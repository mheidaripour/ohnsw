#!/usr/bin/env bash
# Non-partitioned baseline built by the PARTITIONED procedure: every node is
# its own partition (--partitioning singleton). No METIS, no K-means -- the
# assignment is the identity, so step 1 is O(n) and needs no vectors.
#
# WHY THIS EXISTS. run_nopart_bf64.sh (--partitioning none) goes through
# emulate_graph/__emulate_graph: a Vertex plus a binary FanOut out-tree per
# node, and a query collects neighbour ids with add_neighbors(ds='SET'), which
# drops ogE (oblivious_graph.py:1842) and re-enters the next node by id through
# the entry_points AVL tree. Every dereference it makes is therefore on a
# single-alias FanOut pointer, whose cost is depth 1 under BOTH pointer schemes
# -- so BOSAM and OSAM+ measure the same there (36,242 vs 36,341 reads/q on
# sift1m), which is a property of that traversal, not of the schemes.
#
# With |P| = 1 every base-layer edge is a cross-partition edge, so a node's
# payload is one OSAM object holding its neighbour ids together with a pointer
# to each neighbour's payload, and a hop is one dereference of a pointer with
# ~in-degree aliases -- the access pattern random_walk/pagerank measure, where
# the r-ary grouping cuts deref depth from log2(d) to log_bf(d).
#
# Measured on laion 1k, 20 queries, bs=4096 bf=64: 160.7 reads/q (BOSAM) vs
# 470.5 (OSAM+), a 2.9x gap, at identical recall and identical rounds/query.
#
# NOT COMPARABLE TO run_nopart_bf64.sh IN TWO COLUMNS:
#   * Upper layers. use_node_clustering forces levels > 0 to plaintext
#     (o_hnsw.py _get_level_neighbors), as the METIS runs already do. The
#     --partitioning none runs keep them oblivious.
#   * rounds/query. The none path calls increment_search_round() twice per node
#     expansion; the clustered path calls it once per payload fetch.
#
# MEMORY. Should be far below run_nopart_bf64_big.sh: the binary FanOut
# out-tree (~2M-1 SAM cells per node) is gone, leaving one payload plus
# in-degree aliases. tripclick, which does not fit under --partitioning none,
# is worth attempting here. The RSS watchdog traces but does not kill by
# default; see RSS_LIMIT_KB below.
#
# USAGE. With no arguments every dataset runs. Name datasets to run a subset,
# which is how the work is split across machines:
#
#     ./run_singleton_bf64.sh laion1m sift1m        # one box
#     ./run_singleton_bf64.sh msmarco1m tripclick   # the other
#
# Both write the same logs/singleton_bf64/<dataset>_bssweep/ layout, so rsync
# one tree onto the other and point make_tables.py at the merged root.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")" || exit 1
PY=${PY:-/home/mahdieh/miniconda3/envs/ann_osam/bin/python}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-64}
ROOT=${ROOT:-logs/singleton_bf64}
BS=${BS:-4096}
RSS_INTERVAL=${RSS_INTERVAL:-15}

# Trace VmRSS but never kill: RSS_LIMIT_KB defaults to 0 (watchdog disabled).
# The .rsslog still records the whole curve and the peak, so a run that dies to
# the OOM killer or drives the box into swap is still diagnosable after the
# fact -- we just do not pre-empt it. To arm the killer, set a limit
# explicitly, e.g. 95% of MemTotal:
#
#     RSS_LIMIT_KB=$(( $(awk '/^MemTotal:/{print $2}' /proc/meminfo) * 95 / 100 )) \
#         ./run_singleton_bf64.sh msmarco1m tripclick
RSS_LIMIT_KB=${RSS_LIMIT_KB:-0}

# Log dir names are what make_tables globs: <dataset>_bssweep, except sift1m,
# whose DatasetSpec.log_dir pins "sift1m-l2_bssweep" (make_tables.py:87).
# Ordered cheapest-first so a mistake surfaces before the big runs start.
CONFIGS=$(cat <<'TSV'
laion1m    laion1m_bssweep    ip  1000  data/laion/1m/hnsw_32_40_ip.json        data/laion/1m/laion_base.fvecs  data/laion/laion_query.fvecs  data/laion/1m/gt.ivecs  data/laion/1m/hnsw_32_40_ip.index
sift1m     sift1m-l2_bssweep  l2  1000  data/sift/hnsw_64_80_2_ip.json          data/sift/base.fvecs            data/sift/query.fvecs         data/sift/data/dataset/sift/gt.ivecs  data/sift/data/dataset/sift/hnsw_64_80_2_ip.index
msmarco1m  msmarco1m_bssweep  ip  1000  data/msmarco/hnsw_128_200_2_ip_1m.json  data/msmarco/data/dataset/msmarco_bert/passages_1m.fvecs  data/msmarco/data/dataset/msmarco_bert/queries.fvecs  data/msmarco/data/dataset/msmarco_bert/gt_10_1m.ivecs  data/msmarco/data/dataset/msmarco_bert/hnsw_128_200_2_ip_1m.index
tripclick  tripclick_bssweep  ip  1000  data/tripclick/hnsw_128_160_2_ip.json   data/tripclick/data/dataset/trip_distilbert/passages.fvecs  data/tripclick/data/dataset/trip_distilbert/queries.fvecs  data/tripclick/data/dataset/trip_distilbert/gt_10.ivecs  data/tripclick/data/dataset/trip_distilbert/hnsw_128_160_2_ip.index
TSV
)

# Optional dataset filter from the command line; empty means all of CONFIGS.
# A typo must not silently run nothing, so unknown names are a hard error.
SELECT=("$@")
KNOWN=$(awk 'NF {print $1}' <<< "$CONFIGS")
for want in "${SELECT[@]:-}"; do
    [ -n "$want" ] || continue
    grep -qx -- "$want" <<< "$KNOWN" || {
        echo "! unknown dataset '$want'; known: $(tr '\n' ' ' <<< "$KNOWN")" >&2
        exit 2
    }
done
selected() {
    (( ${#SELECT[@]} == 0 )) && return 0
    local d
    for d in "${SELECT[@]}"; do [ "$d" = "$1" ] && return 0; done
    return 1
}
if (( ${#SELECT[@]} )); then
    echo "Datasets: ${SELECT[*]}"
else
    echo "Datasets: $(tr '\n' ' ' <<< "$KNOWN") (all)"
fi

# Sample VmRSS until the child exits, killing it past the limit if one is set
# (limit 0, the default, only traces). The peak line
# is written from a TERM trap too, because the watcher is killed as soon as the
# run finishes and that almost always lands inside the sleep.
watch_rss() {
    local pid=$1 out=$2 limit=$3 peak=0 rss
    trap 'printf "peak %s kB\n" "$peak" >> "$out"; trap - TERM; exit 0' TERM
    while kill -0 "$pid" 2>/dev/null; do
        rss=$(awk '/^VmRSS:/{print $2}' "/proc/$pid/status" 2>/dev/null) || break
        [ -n "$rss" ] || break
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

if (( RSS_LIMIT_KB > 0 )); then
    echo "RSS limit: ${RSS_LIMIT_KB} kB  (kill ENABLED)"
else
    echo "RSS limit: none  (tracing only, watchdog will not kill; set RSS_LIMIT_KB to arm it)"
fi
fail=0
while read -r ds logdir metric q graph vectors queries gt index; do
    [ -n "${ds:-}" ] || continue
    selected "$ds" || continue
    D=$ROOT/$logdir; mkdir -p "$D"
    # r-ary first: it is the memory-risky one, so a dataset that cannot fit
    # fails before its multiwrite twin has spent any time.
    for pt in multiwriterary multiwrite; do
        [ "$pt" = multiwriterary ] && s=rary || s=multiwrite
        log=$D/o_hnsw_singleton_${ds}_${s}_bs${BS}.log
        # .rsslog, not .log: the plot scripts glob "*.log" over this directory.
        rsslog=$D/rss_${s}_bs${BS}.rsslog
        if grep -q '^Whole-run allocations:' "$log" 2>/dev/null; then
            echo "skip  $ds $s (complete)"; continue
        fi
        echo "=== $ds $s bs=$BS  $(date '+%F %T')"
        : > "$rsslog"; t0=$(date +%s)
        "$PY" -u o_hnsw.py \
            --graph "$graph" --vectors "$vectors" --queries "$queries" \
            --gt "$gt" --faiss-index "$index" \
            --metric "$metric" --q "$q" --k 10 --ef 64 \
            --partitioning singleton --branching-factor 2 \
            --pointer-type "$pt" --bs "$BS" \
            --pointer-branching-factor 64 \
            --path-oram-rary-read-evictions 2 \
            > "$log" 2>&1 &
        pid=$!
        watch_rss "$pid" "$rsslog" "$RSS_LIMIT_KB" & wpid=$!
        wait "$pid"; rc=$?
        kill "$wpid" 2>/dev/null; wait "$wpid" 2>/dev/null
        if grep -q '^Whole-run allocations:' "$log"; then
            echo "    done in $(( $(date +%s) - t0 ))s  (peak $(awk '/^peak/{print int($2/1048576)" GB"}' "$rsslog" | tail -1))"
        else
            echo "    FAILED (exit $rc, $(( $(date +%s) - t0 ))s) -- see $log and $rsslog" >&2
            grep -q KILLED "$rsslog" && echo "    ^ killed by the RSS watchdog" >&2
            fail=$(( fail + 1 ))
        fi
    done
done <<< "$CONFIGS"

echo
printf '%-11s %-11s %-11s %-12s %-14s %s\n' dataset scheme reads/q writes/q allocs R@10
while read -r ds logdir _rest; do
    [ -n "${ds:-}" ] || continue
    selected "$ds" || continue
    for s in rary multiwrite; do
        f=$ROOT/$logdir/o_hnsw_singleton_${ds}_${s}_bs${BS}.log
        [ -f "$f" ] || continue
        printf '%-11s %-11s %-11s %-12s %-14s %s\n' "$ds" "$s" \
            "$(awk -F: '/^Average reads.query:/{gsub(/ /,"",$2);print $2}' "$f")" \
            "$(awk -F: '/^Average writes.query:/{gsub(/ /,"",$2);print $2}' "$f")" \
            "$(awk -F: '/^Whole-run allocations:/{gsub(/ /,"",$2);print $2}' "$f")" \
            "$(awk '/^R@10 =/{print $3}' "$f")"
    done
done <<< "$CONFIGS"

(( fail == 0 )) || echo "! $fail run(s) failed. Peak RSS traces: $ROOT/*/rss_*.rsslog" >&2
exit $(( fail > 0 ))
