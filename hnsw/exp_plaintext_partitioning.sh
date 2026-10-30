#!/usr/bin/env bash
#
# Plaintext partitioning comparison: METIS vs K-means at one operating point.
#
# Runs cluster/plaintext/hnsw-metis-partitioned.py and
# cluster/plaintext/hnsw-clustered.py over the same HNSW graph with the same
# partition count, so their partition-quality and query numbers are directly
# comparable. Feeds the appendix tables built by plots/make_partition_tables.py.
#
#   ./exp_plaintext_partitioning.sh                  # both datasets, both methods
#   ./exp_plaintext_partitioning.sh --dataset sift1m # one dataset
#   ./exp_plaintext_partitioning.sh --method metis   # the cheap half first
#
# Both scripts partition the *same* graph JSON, but along different axes:
# METIS cuts the symmetrized base-layer adjacency, K-means clusters the base
# vectors and ignores the graph entirely. That is the comparison -- see
# paper/appendix_partitioning.tex.
#
# Cost. K-means dominates and scales with the vector dimension: measured on
# this machine at n=100k/k=100/20 iters/n_init=1, sklearn takes 1.27 s at
# d=128 and 5.94 s at d=512, a 4.7x ratio matching 512/128. Scaled to
# n=1M, k=1000, n_init=10 that is roughly 45 min for SIFT-1M and 3-4 h for
# LAION-1M, against ~8 min for METIS on either. Runs are launched in the
# background; LAION K-means is the long pole.
#
# Resumable: a run whose JSON output already exists is skipped unless --force.

set -uo pipefail

cd "$(dirname "$(readlink -f "$0")")" || exit 1

PY=${PY:-/home/mahdieh/miniconda3/envs/ann_osam/bin/python}

# Required, not advisory. This machine reports 192 cores; OpenBLAS is
# precompiled for 64 and aborts with "corrupted size vs. prev_size" when the
# thread count exceeds that. Confirmed by crashing a k-means benchmark with
# OMP_NUM_THREADS=$(nproc) and having it complete with the cap in place.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=64

# The single operating point the appendix reports. SIFT's own block-size sweep
# uses n_c=2000; 1000 is used here so the two datasets are comparable to each
# other, and the appendix says so.
NPARTS=1000
K=10
EF=64
NUM_QUERIES=1000
DATASETS=(sift1m laion1m)
METHODS=(metis kmeans)
FORCE=0
DRY_RUN=0

LOGROOT=cluster/plaintext/log

usage() {
    cat <<'EOF'
Usage: ./exp_plaintext_partitioning.sh [options]

  --dataset NAME     sift1m | laion1m   (repeatable; default: both)
  --method NAME      metis | kmeans     (repeatable; default: both)
  --n-parts N        partition/cluster count (default: 1000)
  --k N              neighbours to retrieve (default: 10)
  --ef N             efSearch (default: 64)
  --num-queries N    queries to run (default: 1000)
  --force            rerun even if an output JSON already exists
  --dry-run          print the commands and exit
  -h, --help         this message
EOF
}

DATASETS_SET=0
METHODS_SET=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            if [[ $DATASETS_SET -eq 0 ]]; then DATASETS=(); DATASETS_SET=1; fi
            DATASETS+=("$2"); shift 2 ;;
        --method)
            if [[ $METHODS_SET -eq 0 ]]; then METHODS=(); METHODS_SET=1; fi
            METHODS+=("$2"); shift 2 ;;
        --n-parts)     NPARTS="$2"; shift 2 ;;
        --k)           K="$2"; shift 2 ;;
        --ef)          EF="$2"; shift 2 ;;
        --num-queries) NUM_QUERIES="$2"; shift 2 ;;
        --force)       FORCE=1; shift ;;
        --dry-run)     DRY_RUN=1; shift ;;
        -h|--help)     usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

# Paths and metrics, all verified against the files on disk.
#
# SIFT is run with l2, not the historical ip. Its graph JSON declares
# metric_type 1 (L2) despite the _ip filename, and exp_blocksize_sweep.sh:141
# records that ip costs ~9% recall before the ANN search does anything. The
# paper's tab_network.tex is likewise built from sift1m-l2_bssweep.
#
# LAION is run with ip, matching its sweep. Its base vectors are unit-norm, so
# minimising L2 and maximising IP rank identically; the choice is cosmetic.
#
# LAION's gt.ivecs holds exactly 1000 queries x k=10. Asking for more queries
# than that silently loses recall, so --num-queries must not exceed 1000.
dataset_config() {
    case "$1" in
        sift1m)
            GRAPH=data/sift/hnsw_64_80_2_ip.json
            VECTORS=data/sift/base.fvecs
            QUERIES=data/sift/query.fvecs
            # The doubly-nested path is real, not a typo.
            GT=data/sift/data/dataset/sift/gt.ivecs
            METRIC=l2
            LABEL="SIFT-1M"
            ;;
        laion1m)
            GRAPH=data/laion/1m/hnsw_32_40_ip.json
            VECTORS=data/laion/1m/laion_base.fvecs
            # 1M has no query set of its own; it shares the one a level up.
            QUERIES=data/laion/laion_query.fvecs
            GT=data/laion/1m/gt.ivecs
            METRIC=ip
            LABEL="LAION-1M"
            ;;
        *)
            echo "unknown dataset: $1" >&2; return 1 ;;
    esac
}

# The two scripts spell the partition count differently and write to different
# directories, but take the same flags otherwise.
method_config() {
    case "$1" in
        metis)
            SCRIPT=cluster/plaintext/hnsw-metis-partitioned.py
            COUNT_FLAG=--n-partitions
            LOGDIR="$LOGROOT/hnsw_metis_partitioned"
            OUT_GLOB="hnsw_metis_partitioned_%s_*.json"
            ;;
        kmeans)
            SCRIPT=cluster/plaintext/hnsw-clustered.py
            COUNT_FLAG=--n-clusters
            LOGDIR="$LOGROOT/hnsw_clustered"
            OUT_GLOB="hnsw_clustered_%s_*.json"
            ;;
        *)
            echo "unknown method: $1" >&2; return 1 ;;
    esac
}

for f in "$LOGROOT/hnsw_metis_partitioned" "$LOGROOT/hnsw_clustered"; do
    mkdir -p "$f"
done

TS=$(date +%Y%m%d_%H%M%S)
launched=0

for ds in "${DATASETS[@]}"; do
    dataset_config "$ds" || exit 1
    for missing in "$GRAPH" "$VECTORS" "$QUERIES" "$GT"; do
        if [[ ! -f "$missing" ]]; then
            echo "SKIP $ds: missing $missing" >&2
            continue 2
        fi
    done

    for method in "${METHODS[@]}"; do
        method_config "$method" || exit 1
        RUN_NAME="${ds}_k${NPARTS}"

        # shellcheck disable=SC2059
        existing=$(compgen -G "$LOGDIR/$(printf "$OUT_GLOB" "$RUN_NAME")" 2>/dev/null | head -1)
        if [[ -n "$existing" && $FORCE -eq 0 ]]; then
            echo "SKIP $ds/$method: $existing already exists (use --force to redo)"
            continue
        fi

        STDOUT_LOG="$LOGDIR/${RUN_NAME}_${TS}.log"
        cmd=("$PY" -u "$SCRIPT"
             --json "$GRAPH"
             --vectors "$VECTORS"
             --queries "$QUERIES"
             --gt "$GT"
             "$COUNT_FLAG" "$NPARTS"
             --metric "$METRIC"
             --k "$K" --ef "$EF"
             --num-queries "$NUM_QUERIES"
             --run-name "$RUN_NAME"
             --write-neighbor-cluster-json)
        # --name is METIS-only: it sets the dataset label in the report header.
        if [[ "$method" == "metis" ]]; then
            cmd+=(--name "$LABEL")
        fi

        if [[ $DRY_RUN -eq 1 ]]; then
            printf '%q ' "${cmd[@]}"; printf '> %s\n' "$STDOUT_LOG"
            continue
        fi

        echo "RUN  $ds/$method (k=$NPARTS, metric=$METRIC) -> $STDOUT_LOG"
        nohup "${cmd[@]}" > "$STDOUT_LOG" 2>&1 &
        echo "     pid $!"
        launched=$((launched + 1))
    done
done

if [[ $DRY_RUN -eq 0 ]]; then
    echo
    echo "launched $launched run(s); JSON lands in $LOGROOT/hnsw_{metis_partitioned,clustered}/"
    echo "next: python plots/make_partition_tables.py --k $NPARTS"
fi
