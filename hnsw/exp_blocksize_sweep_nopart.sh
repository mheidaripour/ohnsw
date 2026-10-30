#!/usr/bin/env bash
#
# Block-size sweep for the NON-partitioned oblivious HNSW (--partitioning none).
#
# The partitioned sweep (exp_blocksize_sweep.sh) walks a partition count and
# derives a block size from the resulting partition layout. There is no
# partition here, so this script walks --bs directly, from 64 bytes up to the
# point where the r-ary reference tree goes flat. Past that point nothing
# changes -- measured on 1k, bs=4096 and bs=8192 produce byte-identical reads,
# writes and allocations -- so the ladder stops there rather than paying for
# duplicate runs.
#
#   python probe_nopart.py --graph data/1k/hnsw_32_40.json --dataset 1k
#   ./exp_blocksize_sweep_nopart.sh --dataset 1k
#
# The cap comes from probe_nopart.py's CSV, the same way the partitioned sweep
# takes its block sizes from probe_partitions.py's:
#
#   bs_copies = next_pow2(8 * (max_in_degree + 1) + 16)
#
# which is where every pointee's reference tree becomes depth-1. Without
# partitioning, SmartPointer.copy is called once per *in-edge* of a vertex
# (oblivious_graph.py:1191-1198), so in-degree is what sets the fan-out needed.
#
# Runs are sequential and resumable: a config whose log reached the memory-access
# summary is skipped, so an interrupted sweep picks up where it stopped. --force
# redoes everything.
#
# ---------------------------------------------------------------------------
# READ THIS BEFORE PLOTTING LATENCY.
#
# The tables this feeds (make_tables.py: tab_datasets, tab_cost, tab_network)
# are meaningful. The latency-vs-block-size FIGURE is not, for this
# configuration, and the reason is arithmetic rather than a missing run.
#
# plot_latency.py models
#     latency = 4*(b+1)*bs*log2(allocs)/BW  +  RTT * reads_per_query
# and the partitioned figure has an interior minimum because growing partitions
# cut reads/query ~24x (414.6 -> 17.2 on 1k), collapsing the second term until
# the first -- linear in bs -- overtakes it.
#
# Without partitioning there is no partition to grow. Measured on 1k:
#
#     bs      reads/q    bandwidth term     RTT term      latency
#     64      14524.2       0.000098 s    1452.42 s    1452.42 s
#     1024    14462.5       0.001564 s    1446.25 s    1446.25 s
#     4096    14461.5       0.006252 s    1446.15 s    1446.16 s
#     8192    14461.5       0.012504 s    1446.15 s    1446.16 s
#
# Latency is 99.9996% RTT term, varies 0.4% end to end, and the two terms only
# cross at a 0.9 GiB block -- 226x past the top of this sweep. The curve is
# monotone, its minimum sits at an endpoint, and plot_latency.py will say so
# rather than draw a ring. For multiwrite it is worse: --bs never reaches that
# code path, so its curve is exactly constant by construction.
#
# That is a finding, not a failure. Report it as one; do not widen the sweep
# hoping for a knee that the model says is three orders of magnitude away.
# ---------------------------------------------------------------------------

set -uo pipefail

cd "$(dirname "$(readlink -f "$0")")" || exit 1

PY=${PY:-/home/mahdieh/miniconda3/envs/ann_osam/bin/python}

# Required on many-core machines, per cluster/plaintext/README_partitioning.md:
# without them OpenBLAS oversubscribes and the process dies with
# "double free or corruption".
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-64}

DATASET=1k
CSV=""
LOGDIR=""
QUERIES_N=""
K=10
EF=64
BRANCHING=2
MIN_BS=64
MAX_BS=""          # empty -> the dataset's bs_copies, read from the probe CSV
JOBS=1
RSS_LIMIT_KB=""
RSS_INTERVAL=15
FORCE=0
DRY_RUN=0
MULTIWRITE_ONCE=0
POINTER_TYPES=(multiwriterary multiwrite)

usage() {
    cat <<'EOF'
Usage: ./exp_blocksize_sweep_nopart.sh [options]

  --dataset NAME     1k | 10k | 100k | laion1m          (default: 1k)
  --csv PATH         probe CSV        (default: sweep/nopart_<dataset>.csv)
  --logdir PATH      log directory    (default: logs/nopart/<dataset>_bssweep)
  --q N              queries per run  (default: 100 for 1k/10k/100k,
                                       1000 for laion1m -- see the note below)
  --k N              neighbours                          (default: 10)
  --ef N             ef_search                           (default: 64)
  --branching-factor N   OGraph out-tree fan-out         (default: 2)
  --min-bs N         smallest block size, bytes          (default: 64)
  --max-bs N         largest block size, bytes  (default: the probe's bs_copies)
  --pointer-types A B    schemes to run   (default: multiwriterary multiwrite)
  --multiwrite-once  run multiwrite at the largest block size only. --bs is
                     inert for it, so every other point is a duplicate run.
                     Saves half the sweep if you only need the tables; the
                     latency figure needs the full series to draw its line.
  --jobs N           parallel runs; >1 refused on laion1m      (default: 1)
  --rss-limit-kb N   kill a run above this RSS  (default: 95% of MemTotal)
                     0 disables the kill but keeps the trace
  --force            re-run configs that already have a complete log
  --dry-run          print the commands without running them
  -h, --help         this message

The log directory is logs/nopart/<dataset>_bssweep so that

    python plots/make_tables.py --datasets 1k 10k 100k laion1m \
        --logs-root logs/nopart --out paper/tables_nopart

finds every dataset without touching the partitioned logs or tables.

NOTE on --q for laion1m: make_tables.DatasetSpec declares 1000 queries, and
check_against_spec warns on any row that disagrees. But a non-partitioned run
costs ~15k reads/query, so 1000 queries at 1M scale runs for hours. --q 100 is
the tractable choice and costs one warning per row.

NOTE: make_tables.py has no "100k" entry in its DATASETS dict. Add

    "100k": DatasetSpec("LAION-100k", 100_000, 512, 40, 64, 32, "IP", 100),

before building tables that include it.
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)           DATASET=$2; shift 2 ;;
        --csv)               CSV=$2; shift 2 ;;
        --logdir)            LOGDIR=$2; shift 2 ;;
        --q)                 QUERIES_N=$2; shift 2 ;;
        --k)                 K=$2; shift 2 ;;
        --ef)                EF=$2; shift 2 ;;
        --branching-factor)  BRANCHING=$2; shift 2 ;;
        --min-bs)            MIN_BS=$2; shift 2 ;;
        --max-bs)            MAX_BS=$2; shift 2 ;;
        --jobs)              JOBS=$2; shift 2 ;;
        --rss-limit-kb)      RSS_LIMIT_KB=$2; shift 2 ;;
        --multiwrite-once)   MULTIWRITE_ONCE=1; shift ;;
        --force)             FORCE=1; shift ;;
        --dry-run)           DRY_RUN=1; shift ;;
        --pointer-types)
            shift
            POINTER_TYPES=()
            while [[ $# -gt 0 && $1 != --* ]]; do POINTER_TYPES+=("$1"); shift; done
            ;;
        -h|--help)           usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

# ---------------------------------------------------------------- dataset ---
# Only the LAION family: this sweep exists for the no-partitioning comparison,
# and those are the four scales it is run at. run.sh is bypassed for the same
# reason exp_blocksize_sweep.sh bypasses it -- it never forwards
# --vectors/--queries/--gt, so it is pinned to the 1k defaults.
#
# Every graph JSON here declares metric_type 1 (L2) except laion/1m's _ip
# variant, and it does not matter: o_hnsw.py never reads that field, and the
# base vectors are unit-norm (measured mean 1.0000, cv 0.024%), so IP and L2
# rank identically. Brute force confirms it -- exact IP reproduces gt's top-10
# at 100% on both 10k and 100k. METRIC is ip throughout, matching every other
# LAION run in this repo.
case $DATASET in
    1k)
        GRAPH=data/1k/hnsw_32_40.json
        VECTORS=data/1k/laion_base.fvecs
        QUERIES=data/1k/laion_query.fvecs
        GT=data/1k/gt.ivecs
        INDEX=data/1k/hnsw_32_40_ip.index
        : "${QUERIES_N:=100}"
        ;;
    10k)
        GRAPH=data/10k/hnsw_32_40.json
        VECTORS=data/10k/laion_base.fvecs
        # 10k has no query set of its own; it shares the top-level one.
        QUERIES=data/laion_query.fvecs
        GT=data/10k/gt.ivecs
        INDEX=data/10k/hnsw_32_40_ip.index
        : "${QUERIES_N:=100}"
        ;;
    100k)
        GRAPH=data/laion/100k/hnsw_32_40.json
        VECTORS=data/laion/100k/laion_base.fvecs
        QUERIES=data/laion/laion_query.fvecs
        GT=data/laion/100k/gt.ivecs
        INDEX=data/laion/100k/hnsw_32_40_ip.index
        : "${QUERIES_N:=100}"
        ;;
    laion1m)
        GRAPH=data/laion/1m/hnsw_32_40_ip.json
        VECTORS=data/laion/1m/laion_base.fvecs
        QUERIES=data/laion/laion_query.fvecs
        GT=data/laion/1m/gt.ivecs
        INDEX=data/laion/1m/hnsw_32_40_ip.index
        : "${QUERIES_N:=1000}"
        ;;
    *)
        echo "unknown dataset: $DATASET (expected 1k, 10k, 100k or laion1m)" >&2
        exit 1
        ;;
esac

METRIC=ip

: "${CSV:=sweep/nopart_${DATASET}.csv}"
: "${LOGDIR:=logs/nopart/${DATASET}_bssweep}"

# Default watchdog: 95% of RAM, and the kill is ACTIVE -- deliberately unlike
# exp_blocksize_sweep.sh, whose kill is commented out.
#
# That script's watchdog fired at 80% of MemTotal, below where the kernel begins
# reclaiming, so it killed 8 tripclick configs at 637-653 GB before a single page
# could reach swap; it was disabled rather than retuned. 95% is the number its
# own comment recommends for a swap-backed run, and it sits above the reclaim
# threshold, so a run that needs swap gets swap and only a run genuinely taking
# the machine down gets killed. That matters more here than there: a
# non-partitioned laion1m build makes ~43.4M pointer aliases, one per edge,
# where the partitioned build only aliases cross-partition edges.
#
# MemTotal is RAM only; swap is not added in, because what is sampled is VmRSS
# (resident pages) and adding swap to a resident-page budget compares two
# different quantities.
if [[ -z $RSS_LIMIT_KB ]]; then
    mem_total=$(awk '/^MemTotal:/{print $2}' /proc/meminfo 2>/dev/null || echo 0)
    RSS_LIMIT_KB=$(( mem_total * 95 / 100 ))
fi

if (( JOBS > 1 )) && [[ $DATASET == laion1m ]]; then
    echo "refusing --jobs $JOBS on laion1m: the non-partitioned build aliases" >&2
    echo "every one of its 43.4M edges, against a 754 GB box where the" >&2
    echo "partitioned sift1m sweep already OOMed at 757 GB. Run sequentially," >&2
    echo "or use --dataset 1k/10k for parallel smoke tests." >&2
    exit 1
fi

for f in "$GRAPH" "$VECTORS" "$QUERIES" "$GT" "$INDEX"; do
    [[ -f $f ]] || { echo "missing input: $f" >&2; exit 1; }
done

# ------------------------------------------------------------- block sizes --
# The cap is the probe's recommended block size: the point where every pointee's
# reference tree is depth-1. Above it reads/writes/allocs are provably identical,
# so running past it buys nothing. Read by header name so the probe's column
# order can change freely, and \r-stripped in case the CSV ever arrives CRLF.
if [[ -z $MAX_BS ]]; then
    [[ -f $CSV ]] || {
        echo "probe CSV not found: $CSV" >&2
        echo "Run:  $PY probe_nopart.py --graph $GRAPH --dataset $DATASET" >&2
        echo "or pass --max-bs explicitly." >&2
        exit 1
    }
    MAX_BS=$(awk -F, '
        { gsub(/\r/, "") }
        NR == 1 { for (i = 1; i <= NF; i++) col[$i] = i; next }
        !("recommended" in col) { print "csv has no recommended column" > "/dev/stderr"; exit 1 }
        $(col["recommended"]) == 1 { print $(col["block_size_bytes"]); exit }
    ' "$CSV")
    [[ -n $MAX_BS ]] || {
        echo "no row marked recommended=1 in $CSV" >&2
        exit 1
    }
fi

# 32 is the hard floor: below it bf < 2 and SmartPointerMultiWriteRary._check_bf
# raises. Every power of two >= 32 yields an even bf, which _check_bf also wants.
BS_LIST=()
bs=32
while (( bs < MIN_BS )); do bs=$(( bs * 2 )); done
while (( bs <= MAX_BS )); do
    BS_LIST+=( "$bs" )
    bs=$(( bs * 2 ))
done

if (( ${#BS_LIST[@]} == 0 )); then
    echo "empty block-size ladder: --min-bs $MIN_BS > --max-bs $MAX_BS" >&2
    exit 1
fi

mkdir -p "$LOGDIR"

# ------------------------------------------------------------------ helpers --

# Short name used in log filenames, matching logparse.POINTER_ALIASES.
alias_for() {
    case $1 in
        multiwriterary) echo rary ;;
        *)              echo "$1" ;;
    esac
}

# `Whole-run allocations:` is the last of the three fields logparse.RunLog
# considers required, so its presence alone means the log is parseable.
is_complete() { [[ -f $1 ]] && grep -q '^Whole-run allocations:' "$1"; }

log_for() { echo "$LOGDIR/o_hnsw_none_${DATASET}_$(alias_for "$1")_bs${2}.log"; }

# Sample VmRSS until the child exits, and kill it if it crosses the limit. The
# peak line is written from a TERM trap as well as on the normal path, because
# run_one kills this watcher as soon as the run finishes and that almost always
# lands inside the sleep -- without the trap every .rsslog would end without a
# peak line.
watch_rss() {
    local pid=$1 out=$2 limit=$3 peak=0 rss
    trap 'printf "peak %s kB\n" "$peak" >> "$out"; trap - TERM; exit 0' TERM
    while kill -0 "$pid" 2>/dev/null; do
        rss=$(awk '/^VmRSS:/{print $2}' "/proc/$pid/status" 2>/dev/null) || break
        [[ -n $rss ]] || break
        (( rss > peak )) && peak=$rss
        printf '%s %s kB\n' "$(date +%H:%M:%S)" "$rss" >> "$out"
        if (( limit > 0 && rss > limit )); then
            printf '%s KILLED pid %s: %s kB exceeds limit %s kB\n' \
                "$(date +%H:%M:%S)" "$pid" "$rss" "$limit" >> "$out"
            kill -9 "$pid" 2>/dev/null
            break
        fi
        sleep "$RSS_INTERVAL"
    done
    printf 'peak %s kB\n' "$peak" >> "$out"
}

run_one() {
    local pt=$1 bs=$2
    local short log rsslog pid wpid rc t0 t1

    short=$(alias_for "$pt")
    log=$(log_for "$pt" "$bs")
    # .rsslog, not .log: the plot scripts glob "*.log" over this directory and
    # would otherwise try to parse the memory traces as runs.
    rsslog="$LOGDIR/rss_${short}_bs${bs}.rsslog"

    if (( ! FORCE )) && is_complete "$log"; then
        echo "  skip  $short bs=$bs  (complete)"
        return 0
    fi

    local -a cmd=(
        "$PY" -u o_hnsw.py
        --graph       "$GRAPH"
        --vectors     "$VECTORS"
        --queries     "$QUERIES"
        --gt          "$GT"
        --faiss-index "$INDEX"
        --q "$QUERIES_N" --k "$K" --ef "$EF" --metric "$METRIC"
        --partitioning none
        --branching-factor "$BRANCHING"
        --pointer-type "$pt" --bs "$bs"
    )

    if (( DRY_RUN )); then
        printf '  %s > %s 2>&1\n' "${cmd[*]}" "$log"
        return 0
    fi

    echo "  run   $short bs=$bs  -> $log"
    : > "$rsslog"
    t0=$(date +%s)
    "${cmd[@]}" > "$log" 2>&1 &
    pid=$!
    watch_rss "$pid" "$rsslog" "$RSS_LIMIT_KB" &
    wpid=$!
    wait "$pid"; rc=$?
    kill "$wpid" 2>/dev/null; wait "$wpid" 2>/dev/null
    t1=$(date +%s)

    if is_complete "$log"; then
        echo "        done in $(( t1 - t0 ))s"
    else
        echo "        FAILED (exit $rc, $(( t1 - t0 ))s) -- see $log and $rsslog" >&2
        return 1
    fi
}

# ------------------------------------------------------------------- sweep --
echo "============================================================"
echo " Non-partitioned block-size sweep: $DATASET"
echo "============================================================"
echo "  graph:        $GRAPH"
echo "  probe CSV:    $CSV"
echo "  block sizes:  ${BS_LIST[*]}"
echo "  cap:          $MAX_BS  (bs_copies -- every reference tree depth-1)"
echo "  pointers:     ${POINTER_TYPES[*]}"
echo "  log dir:      $LOGDIR"
echo "  queries:      $QUERIES_N  (k=$K, ef=$EF, metric=$METRIC, bf=$BRANCHING)"
echo "  RSS limit:    ${RSS_LIMIT_KB} kB  (kill ENABLED; 0 disables)"
echo "============================================================"

failed=0
for pt in "${POINTER_TYPES[@]}"; do
    # r-ary first: it is the memory-risky one, so a config that cannot fit fails
    # before its multiwrite twin has spent any time.
    if (( MULTIWRITE_ONCE )) && [[ $pt == multiwrite ]]; then
        echo ""
        echo "--- $pt (single point: --bs is inert for it)"
        run_one "$pt" "${BS_LIST[-1]}" || failed=$(( failed + 1 ))
        continue
    fi
    for bs in "${BS_LIST[@]}"; do
        echo ""
        echo "--- $pt  bs=$bs"
        run_one "$pt" "$bs" || failed=$(( failed + 1 ))
    done
done

(( DRY_RUN )) && exit 0

# ----------------------------------------------------------------- summary --
echo ""
echo "============================================================"
echo " Summary"
echo "============================================================"
printf '%-12s %-10s %-11s %-11s %-8s %s\n' pointer bs reads/q writes/q R@10 status
for pt in "${POINTER_TYPES[@]}"; do
    for bs in "${BS_LIST[@]}"; do
        short=$(alias_for "$pt")
        log=$(log_for "$pt" "$bs")
        if is_complete "$log"; then
            reads=$(awk -F: '/^Average reads\/query:/{gsub(/ /,"",$2); print $2}' "$log")
            writes=$(awk -F: '/^Average writes\/query:/{gsub(/ /,"",$2); print $2}' "$log")
            r10=$(awk '/^R@10 =/{print $3}' "$log")
            status=ok
        elif (( MULTIWRITE_ONCE )) && [[ $pt == multiwrite && $bs != "${BS_LIST[-1]}" ]]; then
            continue
        else
            reads=-; writes=-; r10=-; status=MISSING
        fi
        printf '%-12s %-10s %-11s %-11s %-8s %s\n' \
            "$short" "$bs" "$reads" "$writes" "$r10" "$status"
    done
done

echo ""
recalls=$(grep -h '^R@10 =' "$LOGDIR"/*.log 2>/dev/null | awk '{print $3}' | sort -u)
if [[ -n $recalls && $(wc -l <<< "$recalls") -eq 1 ]]; then
    echo "✓ recall identical across every run (R@10 = $recalls) -- block size"
    echo "  changes how an address is reached, never which nodes are visited"
else
    echo "! recall varies across runs, so the fixed-accuracy claim does NOT hold:"
    sed 's/^/    R@10 = /' <<< "$recalls"
fi

# The claim the module header makes about multiwrite, checked rather than
# asserted: if --bs really never reaches that code path, every multiwrite run
# in this directory has identical counters.
if [[ " ${POINTER_TYPES[*]} " == *" multiwrite "* ]] && (( ! MULTIWRITE_ONCE )); then
    mw=$(for bs in "${BS_LIST[@]}"; do
             log=$(log_for multiwrite "$bs")
             is_complete "$log" && awk -F: '/^Average reads\/query:/{gsub(/ /,"",$2); print $2}' "$log"
         done | sort -u)
    if [[ -n $mw && $(wc -l <<< "$mw") -eq 1 ]]; then
        echo "✓ multiwrite reads/query identical at every block size ($mw)"
        echo "  -- --bs is inert for it, as expected"
    elif [[ -n $mw ]]; then
        echo "! multiwrite reads/query VARIES with block size:"
        sed 's/^/    /' <<< "$mw"
        echo "  --bs was believed inert for multiwrite (o_hnsw.py:2126 only calls"
        echo "  set_branching_factor for multiwriterary). Investigate before"
        echo "  reporting either curve."
    fi
fi

if (( failed )); then
    echo ""
    echo "! $failed run(s) failed; re-run this script to retry just those."
    exit 1
fi

echo ""
echo "Next:"
echo "  # tables (add a 100k DatasetSpec first if this sweep includes 100k)"
echo "  $PY plots/make_tables.py --datasets 1k 10k 100k laion1m \\"
echo "      --logs-root logs/nopart --out paper/tables_nopart"
echo ""
echo "  # latency figure -- read the header of this script first: the curve is"
echo "  # monotone here and its minimum sits at an endpoint by construction"
echo "  $PY plots/plot_latency.py --logs $LOGDIR --dataset $DATASET"
