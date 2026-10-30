#!/usr/bin/env bash
#
# Block-size sweep under the decoupled PathORAM parameters, for any of the four
# 1M-scale datasets or the 10k laion subset.
#
# o_hnsw.py used to derive the r-ary pointer's branching factor and its per-read
# eviction count from the block size:
#
#     pointer_branching_factor = floor((bs - 16) / 8)
#     rary_read_evictions      = pointer_branching_factor
#
# so every block-size sweep was a branching-factor sweep at the same time, and
# no point on the curve isolated block size. This pins both --
# --pointer-branching-factor 64, --path-oram-rary-read-evictions 1 -- and adds
# the stash replay (--track-stash, 2^24 leaves, bucket Z=4).
#
# That makes the existing logs/<dataset>_bssweep logs INCOMPARABLE with what
# this produces. Nothing here reads or writes those, and the figures are drawn
# over the new directory alone.
#
# Why one script and not four. run_sift1m_blocksize.sh, run_laion1m_blocksize.sh,
# run_msmarco_blocksize.sh and run_tripclick_blocksize.sh are four near-copies
# that have already drifted apart; everything that actually differs between the
# datasets is the case block below, which is the same shape exp_blocksize_sweep.sh
# uses. run_sift1m_newparams.sh predates this and does the sift1m case on its
# own; the two write the same paths, so they must not run at the same time.
#
#   ./run_bssweep_newparams.sh --dataset laion1m --yes
#
# Each dataset's per-run cost is what the corresponding run_*_blocksize.sh
# header documents, plus the stash replay. Rough order: laion1m hours, sift1m
# a day, msmarco1m a day, tripclick the better part of two. All are resumable
# per log, so an interruption costs one config. Run detached:
#
#     nohup ./run_bssweep_newparams.sh --dataset tripclick --yes \
#         > logs/run_tripclick_newparams.log 2>&1 &

set -uo pipefail

cd "$(dirname "$(readlink -f "$0")")" || exit 1

PY=${PY:-/home/mahdieh/miniconda3/envs/ann_osam/bin/python}

# Required on many-core machines, per cluster/plaintext/README_partitioning.md:
# without them OpenBLAS oversubscribes and the process dies with "double free
# or corruption". exp_blocksize_sweep.sh exports these itself; the probe and
# the extractor do not, so they are set here for their benefit.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=64

DATASET=""
PROBE_ONLY=0
SKIP_PROBE=0
SKIP_EXTRACT=0
DRY_RUN=0
ASSUME_YES=0
ORDER=""


TRACK_STASH=1

# The point of this sweep. Changing either of the first two means the runs no
# longer isolate block size, so they are not overridable.
POINTER_BF=64
RARY_READ_EVICTIONS=2

# Replay parameters. Empty means "take the dataset's default from the case
# block": 2^24 leaves for the 1M sets, matching the laion1M run, and 2^18 for
# 10k, which holds leaves-per-node roughly fixed.
#
# The size is not free either way. Too small and the stash never drains; too
# large and the tree is taller than the data needs, so every replayed access
# walks more buckets and the numbers stop meaning what they would at the right
# height. Stash occupancy is only comparable between runs that replayed into
# the same tree, which is why the four 1M sets share one value.
#
# TripClick carries 6x laion's base edges and so a much longer trace; if its
# stash comes out unbounded, a taller tree is the first thing to try:
#     TREE_SIZE=33554432 ./run_bssweep_newparams.sh --dataset tripclick
TREE_SIZE=${TREE_SIZE:-}
BUCKET_CAP=${BUCKET_CAP:-4}

# Latency-model knobs. Post-hoc transforms of the logs, so changing them costs
# nothing and needs no re-run -- see plots/README.md.
BW=${BW:-100e6}
RTT=${RTT:-0.1}

OUT=${OUT:-plots/out_bw100MBps}

# Empty means the dataset's default: 1000 for the 1M sets, 100 for 10k, which
# is what exp_blocksize_sweep.sh uses for each.
Q=${Q:-}

usage() {
    cat <<'EOF'
Usage: ./run_bssweep_newparams.sh --dataset NAME [options]

  --dataset NAME    10k | sift1m | laion1m | msmarco1m | tripclick  (required)
  --probe-only      stop after the probe, before any run time is spent
  --skip-probe      reuse the existing probe CSV as-is
  --skip-extract    assume the graph JSON is current (msmarco1m/tripclick)
  --no-track-stash  leave --track-stash off the o_hnsw.py command line.
                    --path-oram-tree-size and --path-oram-bucket-capacity
                    still go, and o_hnsw.py enables the replay for the tree
                    size alone, so the runs are unchanged.
  --order ORDER     nc-desc (small blocks first) | nc-asc (large blocks first)
                    Default is per dataset; see the case block.
  --dry-run         print the o_hnsw.py commands without running them
  --yes             do not pause for confirmation after the probe
  -h, --help        this message

Environment:
  PY          python to use   (default: the ann_osam conda env)
  NCLUSTERS   partition counts to probe   (default: per dataset)
  LOGDIR      log directory   (default: logs/<dataset>-<metric>_bssweep_np)
  Q           queries per run   (default: 1000; 100 for 10k)
  TREE_SIZE   PathORAM leaves for the replay   (default: per dataset)
  BUCKET_CAP  PathORAM bucket Z                (default: 4)
  BW / RTT    latency model, bytes/s and seconds   (default: 100e6 / 0.1)
  OUT         figure output directory   (default: plots/out_bw100MBps)

The sweep is resumable: a config whose log already reached the memory-access
summary is skipped, so re-running retries only what failed.
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)        DATASET=$2; shift 2 ;;
        --probe-only)     PROBE_ONLY=1; shift ;;
        --skip-probe)     SKIP_PROBE=1; shift ;;
        --skip-extract)   SKIP_EXTRACT=1; shift ;;
        --no-track-stash) TRACK_STASH=0; shift ;;
        --order)          ORDER=$2; shift 2 ;;
        --dry-run)        DRY_RUN=1; shift ;;
        --yes|-y)         ASSUME_YES=1; shift ;;
        -h|--help)        usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

[[ -n $DATASET ]] || { echo "--dataset is required" >&2; usage; exit 1; }

# ---------------------------------------------------------------- dataset ---
# EXTRACT_FROM empty means the graph JSON ships with the dataset. SEEDS are
# existing probe CSVs whose rows are copied in as cache: probe_partitions.py
# keys them on n_clusters and guards them on the graph's size and mtime, so a
# regenerated graph drops every one of them rather than being trusted.
#
# NCLUSTERS extends each dataset's existing ladder downward, because the small
# blocks are the part the old sweeps never reached. Block size is not a free
# parameter: the probe derives it as next_pow2(max_graph_bytes), so a target
# block size is reached by raising the partition count until the largest
# partition's payload fits. Whether the low end materialises is a property of
# the graph -- STEP 2 says plainly which sizes made it.

EXTRACT_FROM=""
EXTRACT=build-linux/extract_hnsw
MINBS=4096
# 8 MiB, not probe_partitions.py's 4 MiB default, so the coarsest rung reports
# its real requirement instead of silently clamping.
MAXBS=8388608

case $DATASET in
    10k)
        # The 10k laion subset. Not a paper dataset -- it is the one that runs
        # in minutes rather than days, so it is where a parameter set gets
        # checked before a 1M sweep is committed to.
        GRAPH=data/10k/hnsw_32_40.json
        VECTORS=data/10k/laion_base.fvecs
        # 10k has no query set of its own; it shares the top-level one.
        QUERIES=data/laion_query.fvecs
        GT=data/10k/gt.ivecs
        INDEX=data/10k/hnsw_32_40_ip.index
        METRIC=ip
        # Existing ladder 2..500 spans 8 KiB .. 1 MiB. nc=1000 lands at 4272 B,
        # which still rounds to 8192, so 2000 and 4000 are the attempt at the
        # 4 KiB floor.
        NCLUSTERS=${NCLUSTERS:-"2 4 8 16 32 63 125 250 500 1000 2000 4000"}
        SEEDS=(sweep/partitions_10k.csv)
        # 1 MiB, not 8: the coarsest rung derives 955476 B and nothing here
        # comes near 4 MiB, so a higher ceiling would only be noise.
        MAXBS=1048576
        : "${ORDER:=nc-desc}"
        # 2^18 leaves against the 1M sets' 2^24, holding leaves-per-node at
        # roughly 26. A 2^24 tree over 10k nodes would be 6 levels taller than
        # the data needs and would make the stash numbers meaningless.
        TREE_SIZE_DEFAULT=262144
        # What exp_blocksize_sweep.sh uses for 10k. The runs are short enough
        # that 1000 is affordable too, if the extra queries are wanted.
        Q_DEFAULT=100
        ;;
    sift1m)
        GRAPH=data/sift/hnsw_64_80_2_ip.json
        VECTORS=data/sift/base.fvecs
        QUERIES=data/sift/query.fvecs
        # The doubly-nested path is real, not a typo.
        GT=data/sift/data/dataset/sift/gt.ivecs
        INDEX=data/sift/data/dataset/sift/hnsw_64_80_2_ip.index
        # L2, not the historical ip. data/sift's ground truth is L2 -- exact L2
        # reproduces gt's top-10 at 100%, exact IP only at 91% -- so an ip run
        # carries a ~9% recall ceiling before the ANN search does anything.
        METRIC=l2
        # 83.8M base edges. Existing ladder 125..32000 reached 32 KiB; 64000
        # and 128000 reached 16K and 8K. 4 KiB is not known to be reachable:
        # nc=256000 gives METIS 247376 partitions whose largest payload is
        # 4824 B, which rounds to 8192. 384000/512000 are the attempt at it.
        NCLUSTERS=${NCLUSTERS:-"125 250 500 1000 2000 4000 8000 16000 32000 64000 128000 256000 384000 512000"}
        SEEDS=(sweep/partitions_sift1m.csv sweep/partitions_sift1m_small.csv)
        # Smallest blocks first: they are the point of this sweep and the
        # cheapest runs, so an interruption costs the large blocks instead.
        : "${ORDER:=nc-desc}"
        # run_sift1m_newparams.sh is the same sweep by another name -- same CSV,
        # same log directory, same parameters -- so the two resume each other
        # but must not run at the same time. Two processes writing one log is
        # not something the resume check can detect.
        if pgrep -f 'run_sift1m_newparams\.sh' > /dev/null 2>&1; then
            echo "! run_sift1m_newparams.sh is running, and it writes the same" >&2
            echo "  logs this would. Wait for it, or pass a different LOGDIR." >&2
            exit 1
        fi
        TREE_SIZE_DEFAULT=16777216
        Q_DEFAULT=1000
        ;;
    laion1m)
        GRAPH=data/laion/1m/hnsw_32_40_ip.json
        VECTORS=data/laion/1m/laion_base.fvecs
        # 1M has no query set of its own -- data/laion/1m holds no
        # laion_query.fvecs, so it shares the one a level up.
        QUERIES=data/laion/laion_query.fvecs
        GT=data/laion/1m/gt.ivecs
        INDEX=data/laion/1m/hnsw_32_40_ip.index
        # Unlike sift1m, either metric is right: the base vectors are unit-norm
        # (mean 1.0000, std 0.0002), and at unit norm minimising L2 and
        # maximising IP are the same ordering. Exact IP and exact L2 each
        # reproduce gt's top-10 at 100%. ip is the historical choice.
        METRIC=ip
        # 43.0M base edges, roughly half sift's, so every rung is halved and
        # the two figures land on the same x axis. This one already reaches
        # 4 KiB: nc=128000 gives 127808 partitions at 3608 B.
        NCLUSTERS=${NCLUSTERS:-"62 125 250 500 1000 2000 4000 8000 16000 32000 64000 128000"}
        SEEDS=(sweep/partitions_laion1m.csv sweep/partitions_laion1m_small.csv)
        : "${ORDER:=nc-desc}"
        TREE_SIZE_DEFAULT=16777216
        Q_DEFAULT=1000
        ;;
    msmarco1m)
        # The first 1,000,000 of MSMARCO's 8,841,823 passages, at Compass's
        # M=128/efC=200. No JSON ships, so it is exported from the index first.
        MS=data/msmarco/data/dataset/msmarco_bert
        GRAPH=data/msmarco/hnsw_128_200_2_ip_1m.json
        EXTRACT_FROM=$MS/hnsw_128_200_2_ip_1m.index
        VECTORS=$MS/passages_1m.fvecs
        QUERIES=$MS/queries.fvecs
        # Scored against the 1M subset, not the full corpus -- a run must be
        # scored against its own base set. gt_10.ivecs is the full-corpus one.
        GT=$MS/gt_10_1m.ivecs
        INDEX=$EXTRACT_FROM
        # Not overridable: the ground truth here is exact inner product by
        # construction (data/msmarco/make_gt.py brute-forces it), so metric and
        # gt cannot disagree the way they do on sift1m.
        METRIC=ip
        # ~169M base edges at d=768, M=128. Existing ladder 250..64000 reached
        # 32 KiB. 128000/256000/512000 are the attempt at 16K/8K/4K; the dense
        # graph makes the 4 KiB end unlikely -- nc=64000 already sits at
        # 29300 B, so halving three times lands near 3700 B at best.
        NCLUSTERS=${NCLUSTERS:-"250 500 1000 2000 4000 8000 16000 32000 64000 128000 256000 512000"}
        SEEDS=(sweep/partitions_msmarco1m.csv)
        # Ascending: memory scales with cross-cluster edges, which grow with
        # partition count, so this front-loads the memory-cheap configs. On a
        # d=768 M=128 graph that ordering has already proved load-bearing --
        # see the tripclick note below.
        : "${ORDER:=nc-asc}"
        TREE_SIZE_DEFAULT=16777216
        Q_DEFAULT=1000
        ;;
    tripclick)
        # The shipped M=128 index. As with msmarco, no JSON ships.
        TRIP=data/tripclick/data/dataset/trip_distilbert
        GRAPH=data/tripclick/hnsw_128_160_2_ip.json
        EXTRACT_FROM=$TRIP/hnsw_128_160_2_ip.index
        VECTORS=$TRIP/passages.fvecs
        QUERIES=$TRIP/queries.fvecs
        # gt_10.ivecs, not bm25_10/obi_*/cluster_trip -- those are other
        # retrieval systems' outputs over the same 1175 topics, not ground truth.
        GT=$TRIP/gt_10.ivecs
        INDEX=$EXTRACT_FROM
        # The one dataset where the metric choice is load-bearing: the vectors
        # are not normalised (norm mean 10.93, std 0.35), and brute force gives
        # exact IP 100% against gt's top-10 but exact L2 only 38%.
        METRIC=ip
        # 257.1M base edges over 1,523,871 nodes -- 6x laion, 3x sift -- so the
        # ladder is laion's x6. Existing 375..96000 reached 32 KiB; 192000 and
        # 384000 are the attempt at 16K and 8K.
        NCLUSTERS=${NCLUSTERS:-"375 750 1500 3000 6000 12000 24000 48000 96000 192000 384000 768000"}
        SEEDS=(sweep/partitions_tripclick.csv)
        # Ascending, and this one is not a preference. At M=128 the previous
        # sweep completed 1 of 9 r-ary configs; the other 8 were killed at
        # 610-653 GB. Memory grows with partition count, so descending order
        # would put the configs most likely to die first and leave nothing.
        : "${ORDER:=nc-asc}"
        TREE_SIZE_DEFAULT=16777216
        Q_DEFAULT=1000
        ;;
    *)
        echo "unknown dataset: $DATASET" >&2
        echo "expected sift1m, laion1m, msmarco1m or tripclick" >&2
        exit 1
        ;;
esac

: "${TREE_SIZE:=$TREE_SIZE_DEFAULT}"
: "${Q:=$Q_DEFAULT}"

case $ORDER in
    nc-desc|nc-asc) ;;
    *) echo "unknown --order: $ORDER (expected nc-desc or nc-asc)" >&2; exit 1 ;;
esac

CSV=sweep/partitions_${DATASET}_np.csv
# Metric in the name because sift1m's new sweep is L2 while its old logs are ip,
# and the directory is the only place that distinction is visible before opening
# a log.
LOGDIR=${LOGDIR:-logs/${DATASET}-${METRIC}_bssweep_np_e2}
TAG=np

START=$(date +%s)

banner() {
    echo ""
    echo "============================================================"
    echo " $* "
    echo " $(date '+%Y-%m-%d %H:%M:%S')   [+$(( ($(date +%s) - START) / 60 )) min]"
    echo "============================================================"
}

# ---------------------------------------------------------------- preflight --
banner "STEP 0/4  Preflight: $DATASET"

# The graph is checked after the export step, not here -- for msmarco1m and
# tripclick it does not exist yet on a first run.
for f in "$VECTORS" "$QUERIES" "$GT" "$INDEX"; do
    [[ -f $f ]] || { echo "✗ missing input: $f" >&2; exit 1; }
done
echo "  ✓ vectors, queries, ground truth and index present"

"$PY" -c 'import pymetis' 2>/dev/null \
    || { echo "✗ pymetis not importable with $PY; the probe cannot run." >&2; exit 1; }
"$PY" -c 'import matplotlib' 2>/dev/null \
    || { echo "✗ matplotlib not importable with $PY; the figures cannot be drawn." >&2; exit 1; }
echo "  ✓ pymetis and matplotlib importable"

mem_free=$(awk '/^MemAvailable:/{print int($2/1048576)}' /proc/meminfo)
echo "  memory available: ${mem_free} GB"
echo "  metric:           $METRIC"
echo "  pointer bf:       $POINTER_BF   (pinned, not derived from block size)"
echo "  r-ary evictions:  $RARY_READ_EVICTIONS"
echo "  stash replay:     ${TREE_SIZE} leaves, bucket Z=${BUCKET_CAP}"
(( TRACK_STASH )) || \
    echo "                    (--track-stash omitted; the tree size enables the"
(( TRACK_STASH )) || \
    echo "                     replay on its own, so this is unchanged)"
echo "  queries:          $Q  (k=10, ef=64)"
echo "  order:            $ORDER"
echo "  probe CSV:        $CSV"
echo "  log dir:          $LOGDIR"
echo "  untouched:        logs/${DATASET}_bssweep, sweep/partitions_${DATASET}.csv"

mkdir -p sweep logs "$OUT"

# ------------------------------------------------------------------ export --
if [[ -z $EXTRACT_FROM ]]; then
    banner "STEP 1/4  Export graph JSON -- not needed ($DATASET ships one)"
elif (( SKIP_EXTRACT )); then
    banner "STEP 1/4  Export graph JSON -- SKIPPED (--skip-extract)"
else
    banner "STEP 1/4  Export the FAISS index to graph JSON"
    # Redone only when the index is newer than the JSON, so re-runs skip it.
    # A few minutes and a multi-GB file: 4.3 GB for tripclick, ~2.8 GB for
    # msmarco1m.
    if [[ -f $GRAPH && $GRAPH -nt $EXTRACT_FROM ]]; then
        echo "  skip: $GRAPH is current ($(stat -c%s "$GRAPH") bytes)"
    else
        [[ -x $EXTRACT ]] || {
            echo "✗ $EXTRACT not found or not executable; build it first." >&2
            exit 1
        }
        # Disk: refuse early rather than dying mid-write.
        avail_kb=$(df -Pk . | awk 'NR==2{print $4}')
        if (( avail_kb < 12 * 1024 * 1024 )); then
            echo "✗ only $(( avail_kb / 1024 / 1024 )) GB free; the export needs ~5 GB" >&2
            echo "  plus room for logs. Free some space first." >&2
            exit 1
        fi
        echo "  running $EXTRACT (this prints the per-layer stats)"
        "$EXTRACT" "$EXTRACT_FROM" "$GRAPH" 2>&1 | tee "logs/extract_hnsw_${DATASET}_np.log"
        rc=${PIPESTATUS[0]}
        if (( rc != 0 )) || [[ ! -s $GRAPH ]]; then
            echo "✗ extract_hnsw failed (exit $rc); nothing downstream can run." >&2
            exit 1
        fi
    fi
fi

[[ -f $GRAPH ]] || { echo "✗ missing graph: $GRAPH" >&2; exit 1; }
echo "  graph: $GRAPH ($(stat -c%s "$GRAPH") bytes)"

# ------------------------------------------------------------------- seed ---
# Done before the --skip-probe branch so a --dry-run --skip-probe has a CSV to
# read. The seeds already carry run flags, so a seeded-but-unprobed CSV
# describes a valid sweep -- just one that stops where the old ladder stopped.
#
# First use only. Re-seeding an existing CSV would undo whatever the last probe
# wrote, including the rows for the new counts.
if [[ ! -f $CSV ]]; then
    echo "  seeding $CSV from ${SEEDS[*]}"
    seeded=0
    for s in "${SEEDS[@]}"; do
        [[ -f $s ]] || { echo "  ! seed not found, skipping: $s"; continue; }
        if (( seeded == 0 )); then
            cat "$s" > "$CSV"
        else
            # Header line dropped: one header, then the data rows.
            tail -n +2 "$s" >> "$CSV"
        fi
        seeded=$(( seeded + 1 ))
    done
    if (( seeded == 0 )); then
        echo "  ! no seeds available; every partition count will be recomputed"
        echo "    from scratch, which takes hours rather than minutes."
    else
        echo "  ✓ seeded $(( $(wc -l < "$CSV") - 1 )) cached row(s)"
    fi
else
    echo "  reusing existing $CSV as the probe cache"
fi

# ------------------------------------------------------------------- probe --
if (( SKIP_PROBE )); then
    banner "STEP 2/4  Probe -- SKIPPED (--skip-probe)"
    [[ -f $CSV ]] || { echo "✗ --skip-probe given but $CSV does not exist" >&2; exit 1; }
else
    banner "STEP 2/4  Probe partition sizes -> block sizes"
    echo "  n_clusters: $NCLUSTERS"
    echo ""
    "$PY" -u probe_partitions.py \
        --graph "$GRAPH" \
        --dataset "$DATASET" \
        --n-clusters $NCLUSTERS \
        --min-bs "$MINBS" \
        --max-bs "$MAXBS" \
        --out "$CSV" \
        2>&1 | tee "logs/probe_${DATASET}_np.log"
    rc=${PIPESTATUS[0]}
    if (( rc != 0 )); then
        echo "✗ probe failed (exit $rc); nothing downstream can run." >&2
        exit 1
    fi
fi

# ------------------------------------------------------------- probe review --
# The things worth stopping for, checked here so they are not missed in the
# probe's own output. None is fatal on its own -- each changes what the sweep is
# worth doing, which is a judgement call, not an error.
banner "STEP 2b/4  Review what the probe derived"

probe_warn=0

echo "  rows marked run=1:"
awk -F, '
    { gsub(/\r/, "") }
    NR == 1 { for (i = 1; i <= NF; i++) c[$i] = i; next }
    $(c["run"]) == 1 {
        printf "    nc=%-8s -> %-8s parts, max %s nodes / %s B -> block %s B%s\n",
            $(c["n_clusters_requested"]), $(c["num_partitions"]),
            $(c["max_partition_nodes"]), $(c["max_graph_bytes"]),
            $(c["block_size_bytes"]),
            ($(c["clamped"]) == 1 ? "   [CLAMPED]" : "")
    }
' "$CSV"

# 1. How far down the block-size range actually reaches. Not every dataset can
#    reach the 4 KiB floor -- a denser graph runs out of partition size first --
#    and that is a result, not a failure.
min_bs_run=$(awk -F, '
    { gsub(/\r/, "") }
    NR == 1 { for (i = 1; i <= NF; i++) c[$i] = i; next }
    $(c["run"]) == 1 { b = $(c["block_size_bytes"]) + 0; if (m == 0 || b < m) m = b }
    END { print m + 0 }
' "$CSV")

if (( min_bs_run <= MINBS )); then
    echo "  ✓ the sweep reaches the ${MINBS} B floor"
else
    echo "  ! the smallest block size in the sweep is ${min_bs_run} B, not ${MINBS} B:"
    echo "    no partition count probed brings the largest partition's payload"
    echo "    under ${MINBS} B. That is a property of this graph. To push further,"
    echo "    re-run with a higher count, e.g."
    echo "        NCLUSTERS=\"$NCLUSTERS $(( $(echo "$NCLUSTERS" | awk '{print $NF}') * 2 ))\" \\"
    echo "            ./run_bssweep_newparams.sh --dataset $DATASET --probe-only"
    echo "    Cached rows are reused, so only the new count costs METIS time."
    probe_warn=$(( probe_warn + 1 ))
fi

# 2. Clamping means the raw block size fell outside [--min-bs, --max-bs]. At the
#    small end that means a single partition's payload already exceeds the
#    floor, so the block cannot hold it and the run does not measure that block
#    size at all.
if awk -F, '
    { gsub(/\r/, "") }
    NR == 1 { for (i = 1; i <= NF; i++) c[$i] = i; next }
    $(c["run"]) == 1 && $(c["clamped"]) == 1 { hit = 1 }
    END { exit !hit }
' "$CSV"; then
    echo "  ! at least one config is CLAMPED: its true block size is outside"
    echo "    [$MINBS, $MAXBS], so that block cannot hold its largest partition."
    echo "    That block size is unreachable on this graph -- drop it."
    probe_warn=$(( probe_warn + 1 ))
fi

# 3. pymetis prints "too many parts!" well before the top of these ladders. A
#    large shortfall means the layout the sweep builds is not the one that was
#    asked for, and the figure's x axis stops describing the runs.
awk -F, '
    { gsub(/\r/, "") }
    NR == 1 { for (i = 1; i <= NF; i++) c[$i] = i; next }
    $(c["run"]) == 1 {
        req = $(c["n_clusters_requested"]); got = $(c["num_partitions"])
        if (got < req * 0.95)
            printf "  ! nc=%s: METIS returned only %s partitions (%.1f%% of requested)\n",
                req, got, 100 * got / req
    }
' "$CSV"

n_configs=$(awk -F, '
    { gsub(/\r/, "") }
    NR == 1 { for (i = 1; i <= NF; i++) c[$i] = i; next }
    $(c["run"]) == 1 { n++ }
    END { print n + 0 }
' "$CSV")

echo ""
echo "  $n_configs block size(s) x 2 pointer types = $(( n_configs * 2 )) runs"
if (( probe_warn )); then
    echo "  $probe_warn warning(s) above -- read them before spending the run time."
else
    echo "  ✓ full ${MINBS} B -> ${MAXBS} B range present, none clamped"
fi

if (( PROBE_ONLY )); then
    banner "Stopping after the probe (--probe-only)"
    echo "When the CSV looks right, continue with:"
    echo "    nohup ./run_bssweep_newparams.sh --dataset $DATASET --skip-probe --yes \\"
    echo "        > logs/run_${DATASET}_newparams.log 2>&1 &"
    exit 0
fi

if (( ! ASSUME_YES )) && [[ -t 0 ]]; then
    echo ""
    read -r -p "Proceed with $(( n_configs * 2 )) runs? [y/N] " reply
    [[ $reply == [yY]* ]] || { echo "aborted."; exit 0; }
fi

# -------------------------------------------------------------------- sweep --
banner "STEP 3/4  Sweep: both pointer types at every block size"

sweep_args=(
    --dataset "$DATASET"
    --csv "$CSV"
    --logdir "$LOGDIR"
    --order "$ORDER"
    --q "$Q" --k 10 --ef 64
    --pointer-types multiwriterary multiwrite
    --pointer-branching-factor "$POINTER_BF"
    --rary-read-evictions "$RARY_READ_EVICTIONS"
    # --path-oram-tree-size "$TREE_SIZE"
    # --path-oram-bucket-capacity "$BUCKET_CAP"
)
# The only thing --no-track-stash removes. The tree size above keeps the replay
# on regardless -- see the TRACK_STASH comment at the top.
# (( TRACK_STASH )) && sweep_args+=(--track-stash)
if (( TRACK_STASH )); then
    sweep_args+=(
        --track-stash
        --path-oram-tree-size "$TREE_SIZE"
        --path-oram-bucket-capacity "$BUCKET_CAP"
    )
fi
(( DRY_RUN )) && sweep_args+=(--dry-run)

METRIC=$METRIC ./exp_blocksize_sweep.sh "${sweep_args[@]}"
sweep_rc=$?

if (( DRY_RUN )); then
    banner "Dry run complete -- nothing was executed"
    exit 0
fi

if (( sweep_rc != 0 )); then
    echo ""
    echo "! Some configs failed. Memory traces are in $LOGDIR/rss_*.rsslog."
    echo "! Continuing to the figures with whatever completed -- re-run with"
    echo "! --skip-probe --skip-extract to retry just the failures."
fi

complete=$(grep -l '^Whole-run allocations:' "$LOGDIR"/*.log 2>/dev/null | wc -l)
if (( complete == 0 )); then
    echo "✗ no complete runs in $LOGDIR; nothing to plot." >&2
    exit 1
fi
echo ""
echo "  $complete complete run(s) in $LOGDIR"

# ------------------------------------------------------------------ figures --
banner "STEP 4/4  Figures"

# Over $LOGDIR alone. The old logs/<dataset>_bssweep runs derived the branching
# factor and the eviction count from their block size and these pin both: same
# x axis, different systems. Merging them the way run_sift1m_smallblocks.sh
# merges its two directories would be wrong here.
echo ""
echo "--- Fig: latency vs block size  (BW=$BW B/s, RTT=$RTT s)"
"$PY" -u plots/plot_latency.py \
    --bandwidth-basis query --bandwidth "$BW" --rtt "$RTT" \
    --logs "$LOGDIR" --dataset "$DATASET" \
    --csv "$OUT/latency_${DATASET}_${TAG}.csv"

"$PY" -u plots/plot_latency.py \
    --bandwidth-basis query --bandwidth "$BW" --rtt "$RTT" \
    --logs "$LOGDIR" --dataset "$DATASET" \
    --paper --out "$OUT/fig_latency_${DATASET}_${TAG}" > /dev/null

# ------------------------------------------------------------------- checks --
banner "Checks worth reading before using these"

# Partitioning changes where data sits, not what the search returns, so a
# varying R@10 means something other than block size moved.
recalls=$(grep -h '^R@10 =' "$LOGDIR"/*.log 2>/dev/null | awk '{print $3}' | sort -u)
if [[ $(wc -l <<< "$recalls") -eq 1 ]]; then
    echo "  ✓ recall identical across every run (R@10 = $recalls)"
else
    echo "  ! recall VARIES across the runs, so the fixed-accuracy premise of"
    echo "    the latency figure does not hold:"
    sed 's/^/      R@10 = /' <<< "$recalls"
fi

# The parameters are the point of this sweep, so confirm the runs actually got
# them rather than trusting the command line.
bad_bf=$(grep -L "Pointer branching:  $POINTER_BF" "$LOGDIR"/*rary*.log 2>/dev/null | wc -l)
bad_ev=$(grep -L "R-ary read evicts:  $RARY_READ_EVICTIONS" "$LOGDIR"/*rary*.log 2>/dev/null | wc -l)
if (( bad_bf == 0 && bad_ev == 0 )); then
    echo "  ✓ every r-ary run reports branching $POINTER_BF, evictions $RARY_READ_EVICTIONS"
else
    echo "  ! $bad_bf r-ary log(s) do not report branching $POINTER_BF and"
    echo "    $bad_ev do not report evictions $RARY_READ_EVICTIONS -- these runs"
    echo "    are not the experiment this script describes."
fi

# A METIS fallback to round-robin destroys locality while still producing a log
# that parses cleanly, so it has to be checked for explicitly.
if grep -lq 'METIS failed' "$LOGDIR"/*.log 2>/dev/null; then
    echo "  ! some runs fell back to round-robin partitioning, which destroys"
    echo "    locality and invalidates them:"
    grep -l 'METIS failed' "$LOGDIR"/*.log | sed 's/^/      /'
else
    echo "  ✓ no METIS fallbacks"
fi

echo ""
echo "  Peak RSS per config:"
grep -H '^peak' "$LOGDIR"/rss_*.rsslog 2>/dev/null \
    | sed 's|.*/rss_||; s|\.rsslog:peak | |' \
    | sort -k2 -n -r | head -6 | sed 's/^/      /'

banner "Done"
echo "Figures:"
ls -1 "$OUT"/fig_latency_${DATASET}_${TAG}.* 2>/dev/null | sed 's/^/  /'
echo ""
echo "Data:"
ls -1 "$OUT"/latency_${DATASET}_${TAG}.csv 2>/dev/null | sed 's/^/  /'
echo ""
echo "  The latency model is a post-hoc transform of the logs, so re-testing"
echo "  network assumptions needs no re-run:"
echo "      $PY plots/plot_latency.py --bandwidth-basis query \\"
echo "          --logs $LOGDIR --bandwidth 100e6"
echo ""
echo "Total elapsed: $(( ($(date +%s) - START) / 60 )) min"
