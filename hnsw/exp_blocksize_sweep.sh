#!/usr/bin/env bash
#
# Block-size sweep: run both pointer types at every partition count.
#
# Reads the CSV that probe_partitions.py produced and, for each row it marked
# run=1, runs o_hnsw.py twice -- once r-ary, once multiwrite -- with identical
# settings apart from --pointer-type. The block size comes from the probe, so
# it always matches the partition layout the run will build.
#
#   python probe_partitions.py --graph data/1k/hnsw_32_40.json --dataset 1k \
#       --n-clusters 2 4 8 16 31 62 125 250
#   ./exp_blocksize_sweep.sh --dataset 1k
#
# Runs are sequential and resumable: a config whose log already reached the
# memory-access summary is skipped, so an interrupted sweep picks up where it
# stopped. Use --force to redo everything.
#
# Memory, not CPU, is the binding constraint on sift1m. One r-ary run has been
# observed climbing to 757 GB RSS on a 754 GB machine before the OOM killer
# took it, and three consecutive attempts at bs=262144 died the same way
# (logs/sift1m/rss_3297006.log, and the truncated *_bs262k_* logs). Every run
# therefore gets an RSS sampler that kills it before it takes the machine with
# it; a killed config costs one config, not the sweep.

set -uo pipefail

cd "$(dirname "$(readlink -f "$0")")" || exit 1

PY=${PY:-/home/mahdieh/miniconda3/envs/ann_osam/bin/python}

# Required on many-core machines, per cluster/plaintext/README_partitioning.md:
# without them OpenBLAS oversubscribes and the process dies with
# "double free or corruption".
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=64

DATASET=1k
CSV=""
LOGDIR=""
QUERIES_N=""
K=10
EF=64
ORDER=nc-desc
JOBS=1
# Empty means "derive from MemTotal" -- see the block below. Note this is a
# plain assignment, not ${RSS_LIMIT_KB:-}, so an inherited environment variable
# of the same name is clobbered here and has no effect. The watchdog is set
# through --rss-limit-kb and nothing else; the run_*_blocksize.sh wrappers do
# not forward it, so raising it means calling this script directly.
RSS_LIMIT_KB=""
RSS_INTERVAL=15
FORCE=0
DRY_RUN=0
POINTER_TYPES=(multiwriterary multiwrite)

# PathORAM knobs, all unset by default so a sweep that does not ask for them
# produces exactly the command line it produced before these existed.
#
# The first two decouple the r-ary pointer from the block size. o_hnsw.py
# derives both from --bs when they are not given -- branching factor
# floor((bs-16)/8), evictions equal to that factor -- which makes a block-size
# sweep a branching-factor sweep at the same time. Pinning them isolates block
# size as the only variable, so a curve over --bs measures what it claims to.
#
# The last three drive the stash replay. They cost run time and change nothing
# the search returns, so they are opt-in.
POINTER_BF=""
RARY_READ_EVICTIONS=""
TRACK_STASH=0
ORAM_TREE_SIZE=""
ORAM_BUCKET_CAP=""

usage() {
    cat <<'EOF'
Usage: ./exp_blocksize_sweep.sh [options]

  --dataset NAME     1k | 10k | sift1m | laion1m | msmarco1m
                     | tripclick | tripclick-m<M>     (default: 1k)
  --csv PATH         probe CSV       (default: sweep/partitions_<dataset>.csv)
  --logdir PATH      log directory   (default: logs/<dataset>_bssweep)
  --q N              queries per run (default: 100 for 1k/10k,
                                      1000 for sift1m/laion1m)
  --k N              neighbours                      (default: 10)
  --ef N             ef_search                       (default: 64)
  --pointer-types A B  pointer types to run
                     (default: multiwriterary multiwrite)
  --order ORDER      nc-desc (small blocks first, fastest runs first)
                     | nc-asc (large blocks first, lowest memory first)
                                                     (default: nc-desc)
  --jobs N           parallel runs; >1 refused outside 1k/10k   (default: 1)
  --rss-limit-kb N   kill a run above this RSS  (default: 80% of MemTotal)
                     0 disables the watchdog
  --force            re-run configs that already have a complete log
  --dry-run          print the commands without running them
  -h, --help         this message

PathORAM knobs, all forwarded to o_hnsw.py only when given:

  --pointer-branching-factor N   r-ary pointer fan-out
                     (default: unset -- o_hnsw.py derives it from --bs)
  --rary-read-evictions N        full PathORAM evictions per r-ary read
                     (default: unset -- o_hnsw.py uses the branching factor)
  --track-stash                  replay the PathORAM trace and print stash
                     stats; requires --path-oram-tree-size
  --path-oram-tree-size N        PathORAM leaves for the replay
  --path-oram-bucket-capacity N  PathORAM bucket Z for the replay

Setting these changes what a run measures, and log filenames do not record
them, so a sweep that uses them writes $LOGDIR/sweep_params.txt and refuses to
run into a directory whose recorded settings differ. Use a fresh --logdir.

Accuracy is not swept. Partitioning changes cost, not search results, so
recall is expected to be identical across every run -- that is what makes the
"fixed accuracy" claim in the figure hold, and the summary at the end checks
it.
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)       DATASET=$2; shift 2 ;;
        --csv)           CSV=$2; shift 2 ;;
        --logdir)        LOGDIR=$2; shift 2 ;;
        --q)             QUERIES_N=$2; shift 2 ;;
        --k)             K=$2; shift 2 ;;
        --ef)            EF=$2; shift 2 ;;
        --order)         ORDER=$2; shift 2 ;;
        --jobs)          JOBS=$2; shift 2 ;;
        --rss-limit-kb)  RSS_LIMIT_KB=$2; shift 2 ;;
        --force)         FORCE=1; shift ;;
        --dry-run)       DRY_RUN=1; shift ;;
        --pointer-branching-factor)   POINTER_BF=$2; shift 2 ;;
        --rary-read-evictions)        RARY_READ_EVICTIONS=$2; shift 2 ;;
        --track-stash)                TRACK_STASH=1; shift ;;
        --path-oram-tree-size)        ORAM_TREE_SIZE=$2; shift 2 ;;
        --path-oram-bucket-capacity)  ORAM_BUCKET_CAP=$2; shift 2 ;;
        --pointer-types)
            shift
            POINTER_TYPES=()
            while [[ $# -gt 0 && $1 != --* ]]; do POINTER_TYPES+=("$1"); shift; done
            ;;
        -h|--help)       usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

# --------------------------------------------------------- PathORAM knobs ---
# Checked here rather than left to o_hnsw.py. argparse would catch all of these
# too, but only inside the first run -- which on sift1m means the sweep prints
# its header, spends minutes loading the graph and partitioning, and only then
# dies on a typo. Every config after it would die the same way.
for pair in "--pointer-branching-factor:$POINTER_BF" \
            "--rary-read-evictions:$RARY_READ_EVICTIONS" \
            "--path-oram-tree-size:$ORAM_TREE_SIZE" \
            "--path-oram-bucket-capacity:$ORAM_BUCKET_CAP"; do
    flag=${pair%%:*}; val=${pair#*:}
    [[ -z $val ]] && continue
    if ! [[ $val =~ ^[1-9][0-9]*$ ]]; then
        echo "$flag must be a positive integer, got: $val" >&2
        exit 1
    fi
done

# Mirrors o_hnsw.py's own check (`--track-stash requires --path-oram-tree-size`),
# for the same reason as above: fail before the run, not inside it.
if (( TRACK_STASH )) && [[ -z $ORAM_TREE_SIZE ]]; then
    echo "--track-stash requires --path-oram-tree-size" >&2
    exit 1
fi

# ---------------------------------------------------------------- dataset ---
# run.sh is deliberately bypassed: it never forwards --vectors/--queries/--gt,
# so it is pinned to the 1k defaults whatever index it is handed. The exp_*.sh
# scripts call o_hnsw.py directly for the same reason.
case $DATASET in
    1k)
        GRAPH=data/1k/hnsw_32_40.json
        VECTORS=data/1k/laion_base.fvecs
        QUERIES=data/1k/laion_query.fvecs
        GT=data/1k/gt.ivecs
        INDEX=data/1k/hnsw_32_40_ip.index
        METRIC=ip
        : "${QUERIES_N:=100}"
        ;;
    10k)
        GRAPH=data/10k/hnsw_32_40.json
        VECTORS=data/10k/laion_base.fvecs
        # 10k has no query set of its own; it shares the top-level one.
        QUERIES=data/laion_query.fvecs
        GT=data/10k/gt.ivecs
        INDEX=data/10k/hnsw_32_40_ip.index
        METRIC=ip
        : "${QUERIES_N:=100}"
        ;;
    sift1m)
        GRAPH=data/sift/hnsw_64_80_2_ip.json
        VECTORS=data/sift/base.fvecs
        QUERIES=data/sift/query.fvecs
        # The doubly-nested path is real, not a typo.
        GT=data/sift/data/dataset/sift/gt.ivecs
        INDEX=data/sift/data/dataset/sift/hnsw_64_80_2_ip.index
        # WARNING: ip is the historical default here, and it is the wrong
        # metric. Brute-forcing 20 queries against the base set shows sift's
        # gt is L2: exact L2 reproduces gt's top-10 at 100%, exact IP only at
        # 91%. The graph agrees -- its JSON declares metric_type 1 (L2),
        # despite the _ip filename. So every ip run carries a ~9% recall
        # ceiling before the ANN search does anything, which is why the
        # sift1m logs plateau at R@10 = 0.921.
        #
        # It half-works because SIFT descriptors have near-constant norm
        # (mean 508.49, CV 0.122%), and at exactly constant norm IP and L2
        # rank identically; the residual variation reshuffles the tightly
        # spaced 10th-11th neighbours.
        #
        # Default left at ip so existing logs stay comparable. For a run that
        # measures the system rather than the metric mismatch:
        #     METRIC=l2 ./exp_blocksize_sweep.sh --dataset sift1m ...
        METRIC=${METRIC:-ip}
        : "${QUERIES_N:=1000}"
        ;;
    laion1m)
        GRAPH=data/laion/1m/hnsw_32_40_ip.json
        VECTORS=data/laion/1m/laion_base.fvecs
        # As with 10k, 1M has no query set of its own -- data/laion/1m holds no
        # laion_query.fvecs, so it shares the one a level up.
        QUERIES=data/laion/laion_query.fvecs
        GT=data/laion/1m/gt.ivecs
        INDEX=data/laion/1m/hnsw_32_40_ip.index
        # The two laion 1M exports share an entry_point, so they are one graph
        # relabelled: hnsw_32_40.json declares metric_type 1 (L2) and
        # hnsw_32_40_ip.json declares 0 (IP). That contradiction does not
        # matter here -- the base vectors are unit-norm (measured: mean 1.0000,
        # std 0.0002), and at unit norm minimising L2 and maximising IP are the
        # same ordering. Brute force confirms it: exact IP and exact L2 each
        # reproduce gt's top-10 at 100%. Unlike sift1m, either choice is right.
        METRIC=${METRIC:-ip}
        : "${QUERIES_N:=1000}"
        ;;
    tripclick|tripclick-m*)
        # `tripclick` is the shipped M=128 index. `tripclick-m<M>` selects a
        # rebuild at another M -- see data/tripclick/hnsw-index.py. M is what
        # sets graph density and therefore the cross-edge count that drives
        # oblivious memory; TRIP_EFC picks the efConstruction the index was
        # built with (recall knob, no memory effect).
        if [[ $DATASET == tripclick ]]; then
            TRIP_M=128
        else
            TRIP_M=${DATASET#tripclick-m}
        fi
        TRIP_EFC=${TRIP_EFC:-160}
        # The JSON is not shipped with the dataset -- run_tripclick_blocksize.sh
        # exports it from the index with build-linux/extract_hnsw first.
        GRAPH=data/tripclick/hnsw_${TRIP_M}_${TRIP_EFC}_2_ip.json
        TRIP=data/tripclick/data/dataset/trip_distilbert
        VECTORS=$TRIP/passages.fvecs
        QUERIES=$TRIP/queries.fvecs
        # gt_10.ivecs, not bm25_10/obi_*/cluster_trip -- those are other
        # retrieval systems' outputs over the same 1175 topics, not ground truth.
        GT=$TRIP/gt_10.ivecs
        INDEX=$TRIP/hnsw_${TRIP_M}_${TRIP_EFC}_2_ip.index
        # Unambiguous here, unlike laion1m/sift1m: the graph declares
        # metric_type 0 (IP) and the index filename agrees.
        METRIC=ip
        # 1175 queries exist; 1000 keeps it comparable with sift1m/laion1m.
        : "${QUERIES_N:=1000}"
        ;;
    msmarco1m)
        # The first 1,000,000 of MSMARCO's 8,841,823 passages, at Compass's
        # M=128/efC=200. The full corpus gets an index (data/msmarco/hnsw-index.py)
        # but not a sweep: at 5.8x TripClick's nodes and the same M it clears
        # 550 GB of fixed cost before the OSAM build starts, against a 603 GB
        # watchdog. run_msmarco_blocksize.sh has the arithmetic.
        MS=data/msmarco/data/dataset/msmarco_bert
        # Not shipped -- run_msmarco_blocksize.sh exports it from the index with
        # build-linux/extract_hnsw first, same as tripclick.
        GRAPH=data/msmarco/hnsw_128_200_2_ip_1m.json
        VECTORS=$MS/passages_1m.fvecs
        QUERIES=$MS/queries.fvecs
        # Scored against the 1M subset, not the full corpus -- a run must be
        # scored against its own base set. gt_10.ivecs is the full-corpus one.
        GT=$MS/gt_10_1m.ivecs
        INDEX=$MS/hnsw_128_200_2_ip_1m.index
        # Unambiguous for a different reason than tripclick: the ground truth
        # here is exact inner product BY CONSTRUCTION (data/msmarco/make_gt.py
        # brute-forces it), so metric and gt cannot disagree the way they do on
        # sift1m. Measured norms are near-constant anyway (passages mean 9.0616,
        # cv 0.571%; queries mean 9.0520, cv 0.849%), so IP and L2 rank almost
        # identically -- closer to sift than to tripclick's cv 3.19%. That makes
        # the choice low-stakes, but ip is the one the gt was built with and the
        # one TCT-ColBERT was trained for, so it is not overridable.
        METRIC=ip
        # 6,980 dev-small queries exist; 1000 keeps it comparable with the others.
        : "${QUERIES_N:=1000}"
        ;;
    *)
        echo "unknown dataset: $DATASET (expected 1k, 10k, sift1m, laion1m, msmarco1m, tripclick or tripclick-m<M>)" >&2
        exit 1
        ;;
esac

: "${CSV:=sweep/partitions_${DATASET}.csv}"
: "${LOGDIR:=logs/${DATASET}_bssweep}"

# Default watchdog: 80% of RAM. The 20% margin is not tuning -- it is what a
# run needs to not take the machine with it. A sift1m run was observed at
# 757 GB on a 754 GB box, at which point the kernel OOM killer picks a victim
# by its own heuristic and that victim is not necessarily this run.
#
# MemTotal is RAM only; swap is deliberately not added in. Two reasons:
#
#   * The measurement below is VmRSS, which counts resident pages. Adding swap
#     to a resident-page budget compares two different quantities.
#   * The kernel does not reclaim into swap until RAM is actually tight, so on
#     a 754 GB box a run sitting at 650 GB is fully resident and this limit
#     fires long before a page ever reaches swap. Adding 509 GB of swap to the
#     machine therefore does nothing on its own -- the swap is unreachable
#     until the limit is raised past where reclaim begins (~95% of RAM).
#
# So a swap-backed run is an explicit choice, not a default:
#     ./exp_blocksize_sweep.sh --rss-limit-kb $(( 754*1024*1024 * 95/100 )) ...
# and what that buys is a slow run instead of a dead one. It also makes every
# .rsslog peak an understatement, since the swapped-out share is invisible to
# VmRSS.
if [[ -z $RSS_LIMIT_KB ]]; then
    mem_total=$(awk '/^MemTotal:/{print $2}' /proc/meminfo 2>/dev/null || echo 0)
    RSS_LIMIT_KB=$(( mem_total * 80 / 100 ))
fi

if (( JOBS > 1 )) && [[ $DATASET == sift1m || $DATASET == laion1m || $DATASET == tripclick* || $DATASET == msmarco* ]]; then
    echo "refusing --jobs $JOBS on $DATASET: a single sift1m run has peaked at" >&2
    echo "757 GB RSS. laion1m measured 53 GB, but tripclick carries 6x laion's" >&2
    echo "base edges and 37 GB of plaintext d=768 vectors on top; msmarco1m is" >&2
    echo "d=768 at M=128, so ~2x sift1m's base edges." >&2
    echo "Run sequentially, or use --dataset 1k/10k for parallel smoke tests." >&2
    exit 1
fi

for f in "$GRAPH" "$VECTORS" "$QUERIES" "$GT" "$INDEX"; do
    [[ -f $f ]] || { echo "missing input: $f" >&2; exit 1; }
done
[[ -f $CSV ]] || {
    echo "probe CSV not found: $CSV" >&2
    echo "Run probe_partitions.py --graph $GRAPH --dataset $DATASET --n-clusters ... first." >&2
    exit 1
}

mkdir -p "$LOGDIR"

# ------------------------------------------------------- log-mixing guard ---
# A log is named o_hnsw_metis_<dataset>_<short>_nc<nc>_bs<bs>.log and records
# nothing about the PathORAM knobs above. Two sweeps that differ only in those
# knobs therefore write the same filenames, and resume makes it worse: the
# second sweep sees a complete log from the first and skips the config, so the
# directory silently ends up holding a mixture of two experiments that the plot
# scripts will draw as one curve.
#
# So the settings are recorded next to the logs and compared on every run. The
# file is written even when every knob is unset, which is what makes an old
# directory (no file, no knobs) distinguishable from a new-parameter one.
#
# Only the knobs are compared. --q/--k/--ef/--metric are left out on purpose:
# they are already visible in every log body, and the plot scripts read them
# from there.
params_file="$LOGDIR/sweep_params.txt"
# The replay's real switch is the tree size, not --track-stash: o_hnsw.py runs
# it for `track_stash or path_oram_tree_size is not None`. Recording the raw
# flag would call two identical sweeps different and refuse to resume one into
# the other, so what goes in the file is the state that flag resolves to.
stash_replay=off
{ (( TRACK_STASH )) || [[ -n $ORAM_TREE_SIZE ]]; } && stash_replay=on
params_now=$(printf 'pointer_branching_factor=%s\nrary_read_evictions=%s\nstash_replay=%s\noram_tree_size=%s\noram_bucket_capacity=%s\n' \
    "${POINTER_BF:-unset}" "${RARY_READ_EVICTIONS:-unset}" "$stash_replay" \
    "${ORAM_TREE_SIZE:-unset}" "${ORAM_BUCKET_CAP:-unset}")

if [[ -f $params_file ]]; then
    if [[ $(cat "$params_file") != "$params_now" ]]; then
        echo "refusing to write into $LOGDIR: it holds runs made with different" >&2
        echo "PathORAM settings, and log filenames cannot tell the two apart." >&2
        echo "" >&2
        echo "  recorded in $params_file:" >&2
        sed 's/^/    /' "$params_file" >&2
        echo "  asked for now:" >&2
        sed 's/^/    /' <<< "$params_now" >&2
        echo "" >&2
        echo "Use a fresh --logdir for this parameter set." >&2
        exit 1
    fi
else
    # No record, but logs already here: they predate this guard, so they were
    # made with the knobs unset. Adding a knob now would mix the two and the
    # file written below would then wrongly certify the whole directory as one
    # experiment. Every sweep that ran before this guard existed lands here.
    knobs_set=0
    [[ -n $POINTER_BF$RARY_READ_EVICTIONS$ORAM_TREE_SIZE$ORAM_BUCKET_CAP ]] && knobs_set=1
    (( TRACK_STASH )) && knobs_set=1
    if (( knobs_set )) && compgen -G "$LOGDIR"/*.log > /dev/null; then
        echo "refusing to write into $LOGDIR: it already holds logs but has no" >&2
        echo "$params_file, so those runs were made with the PathORAM knobs" >&2
        echo "unset -- they cannot be compared with the settings asked for now." >&2
        echo "Use a fresh --logdir for this parameter set." >&2
        exit 1
    fi
    (( DRY_RUN )) || printf '%s\n' "$params_now" > "$params_file"
fi

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

# Sample VmRSS until the child exits, and kill it if it crosses the limit.
#
# This is the thing that kills configs. When a sweep reports FAILED and the log
# stops mid-build, look here first: a watchdog kill leaves a `KILLED pid ...`
# line in that config's .rsslog, and a kernel OOM kill does not (it leaves a
# `Killed process` line in dmesg instead). The two are worth telling apart --
# the first means this limit was too low for that config, the second means the
# machine was genuinely out. Every one of the 8 incomplete tripclick r-ary logs
# is the first kind, killed at 637-653 GB against a 604 GB limit.
#
# One sampler runs per config, spawned in run_one and killed as soon as the run
# returns. A killed config costs one config, not the sweep: the loop continues
# and the config can be retried later, since resume is per-log.
#
# What is measured is VmRSS -- resident pages, not the process's total memory.
# With swap configured those diverge, and everything downstream of this loop
# (the `peak` line, the "Peak RSS per config" summary in the run_* wrappers)
# reports residency rather than footprint. A run that swapped 200 GB looks
# 200 GB smaller here than it was.
#
# limit == 0 disables the kill but not the sampling, so `--rss-limit-kb 0`
# still produces a full trace. Use it to measure what a config actually wants
# before deciding where to put the limit -- on a machine you can afford to lose.
#
# The 15 s interval is a compromise: fine enough to catch the ramp, coarse
# enough that 100+ samples per run do not bloat the .rsslog. It does mean a
# fast allocation spike can overshoot the limit between two samples, which is
# part of why the default leaves 20% of RAM in hand.
#
# The peak line is written from a TERM trap as well as on the normal path:
# run_one kills this watcher as soon as the run finishes, which lands while we
# are almost always inside `sleep $RSS_INTERVAL`. Without the trap the function
# never reaches its final printf, so every .rsslog ends without a `peak` line
# and the "Peak RSS per config" summary comes out empty -- which is exactly
# what happened on the laion1m sweep.
watch_rss() {
    local pid=$1 out=$2 limit=$3 peak=0 rss
    trap 'printf "peak %s kB\n" "$peak" >> "$out"; trap - TERM; exit 0' TERM
    while kill -0 "$pid" 2>/dev/null; do
        rss=$(awk '/^VmRSS:/{print $2}' "/proc/$pid/status" 2>/dev/null) || break
        [[ -n $rss ]] || break
        (( rss > peak )) && peak=$rss
        printf '%s %s kB\n' "$(date +%H:%M:%S)" "$rss" >> "$out"
        # DISABLED 2026-08-12: the kill is commented out so runs can spill into
        # the 509 GB of swap added to this machine. It fired at 80% of RAM
        # (604 GB), which is below where the kernel starts reclaiming, so it
        # killed the 8 tripclick r-ary configs at 637-653 GB before a single
        # page could reach swap -- see the block above `watch_rss`.
        #
        # Sampling and the `peak` line are deliberately left running, so the
        # .rsslog traces and the "Peak RSS per config" summaries are unchanged.
        # What is gone is the backstop: nothing now stops a run from taking the
        # machine to the kernel OOM killer, which picks its own victim and need
        # not pick this run.
        #
        # To restore, uncomment the five lines below. Note `--rss-limit-kb`
        # still parses and still prints in the header, but has no effect while
        # this is commented out.
        # if (( limit > 0 && rss > limit )); then
        #     printf '%s KILLED pid %s: %s kB exceeds limit %s kB\n' \
        #         "$(date +%H:%M:%S)" "$pid" "$rss" "$limit" >> "$out"
        #     kill -9 "$pid" 2>/dev/null
        #     break
        # fi
        sleep "$RSS_INTERVAL"
    done
    printf 'peak %s kB\n' "$peak" >> "$out"
}

run_one() {
    local pt=$1 nc=$2 bs=$3
    local short log rsslog pid wpid rc t0 t1

    short=$(alias_for "$pt")
    # No timestamp: one canonical path per config makes resume an -f test
    # rather than a glob, and makes plot-side dedupe a no-op. Both nc and bs
    # are encoded because several existing logs carry bsXXXk tags that
    # disagree with their own body -- the plot scripts read block size from
    # the log body for exactly that reason.
    log="$LOGDIR/o_hnsw_metis_${DATASET}_${short}_nc${nc}_bs${bs}.log"
    # .rsslog, not .log: the plot scripts glob "*.log" over this directory and
    # would otherwise try to parse the memory traces as runs.
    rsslog="$LOGDIR/rss_${short}_nc${nc}_bs${bs}.rsslog"

    if (( ! FORCE )) && is_complete "$log"; then
        echo "  skip  $short nc=$nc bs=$bs  (complete)"
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
        --partitioning metis --n-clusters "$nc"
        --pointer-type "$pt" --bs "$bs"
    )

    # Appended only when set, so an unset knob leaves the flag off the command
    # line entirely and o_hnsw.py applies its own default. Passing an empty
    # string instead would make argparse fail on a missing value.
    [[ -n $POINTER_BF ]]          && cmd+=(--pointer-branching-factor "$POINTER_BF")
    [[ -n $RARY_READ_EVICTIONS ]] && cmd+=(--path-oram-rary-read-evictions "$RARY_READ_EVICTIONS")
    (( TRACK_STASH ))             && cmd+=(--track-stash)
    [[ -n $ORAM_TREE_SIZE ]]      && cmd+=(--path-oram-tree-size "$ORAM_TREE_SIZE")
    [[ -n $ORAM_BUCKET_CAP ]]     && cmd+=(--path-oram-bucket-capacity "$ORAM_BUCKET_CAP")

    if (( DRY_RUN )); then
        printf '  %s > %s 2>&1\n' "${cmd[*]}" "$log"
        return 0
    fi

    echo "  run   $short nc=$nc bs=$bs  -> $log"
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
# Read by column name so the probe's column order can change freely. The \r
# strip guards against a CRLF CSV, where the last header cell would read
# "run\r" and silently match no column -- the sweep would report nothing to do.
mapfile -t ROWS < <(
    awk -F, '
        { gsub(/\r/, "") }
        NR == 1 { for (i = 1; i <= NF; i++) col[$i] = i; next }
        !("run" in col) { print "csv has no run column" > "/dev/stderr"; exit 1 }
        $(col["run"]) == 1 { print $(col["n_clusters_requested"]) "," $(col["block_size_bytes"]) }
    ' "$CSV" | sort -t, -k1,1n
)

if (( ${#ROWS[@]} == 0 )); then
    echo "no rows marked run=1 in $CSV" >&2
    exit 1
fi

case $ORDER in
    nc-desc) mapfile -t ROWS < <(printf '%s\n' "${ROWS[@]}" | tac) ;;
    nc-asc)  ;;
    *) echo "unknown --order: $ORDER (expected nc-desc or nc-asc)" >&2; exit 1 ;;
esac

echo "============================================================"
echo " Block-size sweep: $DATASET"
echo "============================================================"
echo "  probe CSV:    $CSV"
echo "  configs:      ${#ROWS[@]} x ${#POINTER_TYPES[@]} pointer types"
echo "  log dir:      $LOGDIR"
echo "  queries:      $QUERIES_N  (k=$K, ef=$EF, metric=$METRIC)"
echo "  order:        $ORDER"
echo "  pointer bf:   ${POINTER_BF:-derived from block size}"
echo "  rary evicts:  ${RARY_READ_EVICTIONS:-equal to pointer bf}"
if (( TRACK_STASH )) || [[ -n $ORAM_TREE_SIZE ]]; then
    echo "  stash replay: enabled (leaves=${ORAM_TREE_SIZE}, Z=${ORAM_BUCKET_CAP:-4})"
else
    echo "  stash replay: disabled"
fi
echo "  RSS limit:    ${RSS_LIMIT_KB} kB  (NOT ENFORCED -- kill commented out in watch_rss)"
echo "============================================================"

failed=0
for row in "${ROWS[@]}"; do
    nc=${row%%,*}
    bs=${row##*,}
    echo ""
    echo "--- n_clusters=$nc  block_size=$bs"
    for pt in "${POINTER_TYPES[@]}"; do
        # r-ary first: it is the memory-risky one, so a config that cannot fit
        # fails before its multiwrite twin has spent any time.
        run_one "$pt" "$nc" "$bs" || failed=$(( failed + 1 ))
    done
done

(( DRY_RUN )) && exit 0

# ----------------------------------------------------------------- summary --
echo ""
echo "============================================================"
echo " Summary"
echo "============================================================"
printf '%-10s %-8s %-10s %-9s %-8s %s\n' pointer nc bs reads/q R@10 status
for row in "${ROWS[@]}"; do
    nc=${row%%,*}; bs=${row##*,}
    for pt in "${POINTER_TYPES[@]}"; do
        short=$(alias_for "$pt")
        log="$LOGDIR/o_hnsw_metis_${DATASET}_${short}_nc${nc}_bs${bs}.log"
        if is_complete "$log"; then
            reads=$(awk -F: '/^Average reads\/query:/{gsub(/ /,"",$2); print $2}' "$log")
            r10=$(awk '/^R@10 =/{print $3}' "$log")
            status=ok
            # METIS falling back to round-robin destroys locality, which
            # invalidates the run while still producing a complete-looking log.
            grep -q 'METIS failed' "$log" && status='ROUND-ROBIN FALLBACK'
        else
            reads=-; r10=-; status=MISSING
        fi
        printf '%-10s %-8s %-10s %-9s %-8s %s\n' "$short" "$nc" "$bs" "$reads" "$r10" "$status"
    done
done

echo ""
recalls=$(grep -h '^R@10 =' "$LOGDIR"/*.log 2>/dev/null | awk '{print $3}' | sort -u)
if [[ $(wc -l <<< "$recalls") -eq 1 ]]; then
    echo "✓ recall identical across every run (R@10 = $recalls) -- accuracy is fixed"
else
    echo "! recall varies across runs, so the fixed-accuracy claim does NOT hold:"
    sed 's/^/    R@10 = /' <<< "$recalls"
fi

if (( failed )); then
    echo "! $failed run(s) failed; re-run this script to retry just those."
    exit 1
fi
echo ""
echo "Next:"
echo "  python plots/plot_latency.py  --logs $LOGDIR --dataset $DATASET"
echo "  python plots/plot_tradeoff.py --logs $LOGDIR --pointer-type rary --dataset $DATASET"
