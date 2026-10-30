# Plaintext HNSW Partitioning

Partitions an HNSW base layer (L0) into `k` subgraphs and measures query
behaviour with per-partition subgraph caching. Two methods:

- `hnsw-metis-partitioned.py` — METIS balanced **graph** partitioning
  (minimizes edge cuts, tightly balanced sizes).
- `hnsw-clustered.py` — K-means **vector** clustering (groups by vector
  proximity; sizes are unbalanced).

Both traverse the same HNSW graph with the same search algorithm, so their
partition-quality and query metrics are directly comparable. Their console
reports are not byte-identical — see [Output](#output).

Both read a JSON graph produced by `extract_hnsw`
(see [Converting a FAISS HNSW Index to JSON](../../README_extract_hnsw.md)).

## Prerequisites

```bash
pip install -r /requirements.txt   # numpy, scikit-learn, pymetis
```

`pymetis` needs `libmetis`; only `hnsw-metis-partitioned.py` requires it.

## Running

Run from `osam/hnsw/` — the `data/...` paths below are relative to that
directory, while the scripts live in `cluster/plaintext/`. `-u` keeps the log
unbuffered so progress is visible and survives a crash.

The thread limits are **required**, not advisory. `OMP_NUM_THREADS` must not
exceed the thread count OpenBLAS was compiled for, which is 64 in this
environment; this machine reports 192 cores, and running with
`OMP_NUM_THREADS=$(nproc)` aborts partway through K-means with
`corrupted size vs. prev_size` (older builds report `double free or
corruption`). Setting `OPENBLAS_NUM_THREADS=1` keeps BLAS single-threaded and
leaves the parallelism to scikit-learn's OpenMP loops, which is where it helps.

To run the METIS-vs-K-means comparison the appendix reports, use the driver
rather than the commands below, which it wraps with the right paths, metrics
and thread limits for both datasets:

```bash
./exp_plaintext_partitioning.sh                    # both datasets, both methods
./exp_plaintext_partitioning.sh --dataset sift1m --method metis
```

METIS on SIFT-1M:

```bash
mkdir -p cluster/plaintext/log/hnsw_metis_partitioned

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=64 \
nohup python -u cluster/plaintext/hnsw-metis-partitioned.py \
  --json data/sift/hnsw_64_80_2_ip.json \
  --vectors data/sift/base.fvecs \
  --queries data/sift/query.fvecs \
  --gt data/sift/data/dataset/sift/gt.ivecs \
  --n-partitions 1000 \
  --metric l2 --k 10 --ef 64 \
  --num-queries 1000 \
  --run-name sift1m \
  --write-neighbor-cluster-json \
  > cluster/plaintext/log/hnsw_metis_partitioned/sift1m_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

K-means on SIFT-1M (same flags, `--n-clusters` instead of
`--n-partitions`):

```bash
mkdir -p cluster/plaintext/log/hnsw_clustered

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=64 \
nohup python -u cluster/plaintext/hnsw-clustered.py \
  --json data/sift/hnsw_64_80_2_ip.json \
  --vectors data/sift/base.fvecs \
  --queries data/sift/query.fvecs \
  --gt data/sift/data/dataset/sift/gt.ivecs \
  --n-clusters 1000 \
  --metric l2 --k 10 --ef 64 \
  --num-queries 1000 \
  --run-name sift1m \
  --write-neighbor-cluster-json \
  > cluster/plaintext/log/hnsw_clustered/sift1m_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

For LAION-1M, substitute:

```
  --json data/1m/hnsw_32_40.json \
  --vectors data/1m/laion_base.fvecs \
  --queries data/laion_query.fvecs \
  --gt data/1m/gt.ivecs \
```

`--run-all` ignores the path flags and runs a built-in dataset list
(1k, 10k, 100k, 1M) with preset partition counts, skipping any whose files
are missing.

## Options

| Flag | Default | Notes |
|---|---|---|
| `--n-partitions` (METIS) | `sqrt(ntotal)`, **capped at 100** | pass `1000` explicitly for a 1M index |
| `--n-clusters` (K-means) | `sqrt(ntotal)`, uncapped | `1000` for a 1M index |
| `--metric` | `l2` | `l2` or `ip`; check the graph's `meta.metric_type` (`0`=IP, `1`=L2) rather than trusting the filename |
| `--k` / `--ef` | `10` / `64` | search parameters |
| `--num-queries` | `100` | |
| `--log-dir` | `cluster/plaintext/log/hnsw_{metis_partitioned,clustered}` | resolved next to the script, not the working directory |
| `--run-name` | `single` (or `all`) | appears in output filenames |
| `--name` (METIS only) | empty | dataset label in the report header |
| `--write-neighbor-cluster-json` | off | per-partition neighbor lists (large) |

The two defaults differ deliberately: METIS caps at 100 partitions, K-means
does not. Pass the count explicitly whenever the two are being compared.

## Output

Both scripts print partition quality (edge cuts, cut ratio, intra-edge
fraction, Gini, CV, size percentiles), neighbor-partition counts, the HNSW
layer table, and query stats (latency, partitions accessed, cache hit rate,
hops, recall).

They differ in two places:

- **`hnsw-metis-partitioned.py`** prints a per-layer hop breakdown
  (avg/min/max/total per HNSW layer) and a component table splitting total
  hop cost into partition-search and within-partition search.
  `hnsw-clustered.py` reports hops only as a total and a per-query average.
- **`hnsw-clustered.py`** prints per-cluster min/mean/max for nodes, intra
  edges, cross edges and neighbor clusters, plus estimated cluster sizes in
  bytes, and writes an extra JSON file for them. METIS prints a shorter
  cluster block with no byte estimates.

JSON files written to `--log-dir`:

| File | METIS | K-means |
|---|---|---|
| `hnsw_{metis_partitioned,clustered}_{run_name}_{ts}.json` — query results + partition metrics | yes | yes |
| `hnsw_clustered_cluster_stats_{run_name}_{ts}.json` — per-cluster records | — | yes |
| `hnsw_{metis,clustered}_neighbor_clusters_{run_name}_{ts}.json` — neighbor lists | opt-in | opt-in |

`--run-all` instead writes a single `*_all_{ts}.json` /
`*_all_results_{ts}.json` covering every dataset.

Size estimates (K-means only) assume 4 bytes per integer:
`graph_bytes = 4 * (n_nodes + intra_edges + 2 * cross_edges)`, where cross
edges cost two ints each (neighbor id + owning cluster id).
`graph_plus_vectors_bytes` adds `4 * d * n_nodes` for float32 vectors.

## Notes

- K-means is far slower than METIS, and the gap widens with the vector
  dimension. Its cost scales as `n · k · d · iterations · n_init`; METIS
  never looks at the vectors, so `d` does not enter its cost at all.
  Measured at `n=100k`, `k=100`, 20 iterations, `n_init=1`: sklearn takes
  1.27 s at `d=128` and 5.94 s at `d=512`, a ratio of 4.7× that tracks
  512/128 exactly. `n_init=10` is a further flat 10× — it reruns Lloyd
  from ten different seeds and keeps the lowest-inertia result.

  Measured end to end at `n_c=1000` on the 1M indexes:

  | | SIFT-1M (`d=128`) | LAION-1M (`d=512`) |
  |---|---|---|
  | METIS | 286 s | 134 s |
  | K-means | 1559 s (26 min) | 8062 s (2.2 h) |

  The K-means ratio between the two datasets is 5.2×, close to the 4.7×
  the dimension scaling predicts. METIS is *faster* on LAION despite the
  higher dimension, because it never reads the vectors — it is the
  sparser graph (`M=32` against SIFT's `M=64`) that decides its cost.

  (An earlier revision of this file claimed ~13.6 h for K-means on
  LAION-1M against a ~1.4× gap on SIFT-1M. Those two numbers cannot both
  be right — the dimension ratio is 4.7×, not 54× — and no log survives
  to support either. The measured value is 2.2 h.)
- Recall, hops, and nodes visited are **invariant** to the partitioning
  method on the same index — search traverses the same HNSW graph, and
  partitioning only changes which subgraphs get loaded and cached. At
  `n_c=1000` recall matches exactly on both 1M datasets, and so do hops
  and nodes on SIFT-1M. On LAION-1M hops differ in the fourth
  significant figure (1501.93 vs 1502.00, 0.005%): equidistant
  candidates get broken in enumeration order, which partitioning
  reshuffles. Treat a difference in the leading digits as a bug in the
  search path; a difference this size is expected.
- Wall-clock/QPS numbers are sensitive to machine load; compare
  partitions-accessed and cache hit rate instead.
