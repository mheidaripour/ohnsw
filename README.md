# Block Oblivious Single Access Machines and Oblivious Graph-Based Approximate Nearest Neighbor Search with Constant Client Storage

Research artifact accompanying our USENIX Security '27 submission.

This repository builds on the OSAM+ codebase — the Oblivious Single Access
Machine, its pointer schemes, and the oblivious data structures over them. On
top of it, this work contributes:

- **BOSAM**, a blocked pointer type. It replaces OSAM+'s binary
  incoming-pointer tree with an r-ary one whose fan-out is derived from the
  block size, so a whole sibling group fits in a single block and resolving a
  shared pointer costs a constant number of reads rather than one per level.
- **Backend support for it**, in the SAM layer: writes are buffered and
  released into a read-triggered PathORAM eviction batch, and a stash replay
  reports the client storage a run actually needs.
- **Oblivious HNSW**, a graph-based nearest-neighbour index built on OSAM+ and
  BOSAM, with the graph partitioned so that each partition is one
  oblivious object holding the nodes, its intra-partition adjacency, and its
  pointers to neighbouring partitions.

OSAM+'s graph-algorithm benchmarks are retained and used to compare the two
pointer schemes on synthetic and real graphs. Read, write, and allocation counts
come from instrumentation inside the SAM backend.

## Repository layout

| Path | Contents |
| --- | --- |
| [`python-implementation/`](python-implementation/) | The OSAM+ package and this work's additions to it. `single_access_machine.py` handles address allocation, access accounting, read-triggered eviction, and PathORAM stash replay; `smart_pointer_multi_write_rary.py` is the BOSAM pointer; `oblivious_graph.py` and `smart_{queue,stack,avl_tree}.py` are the oblivious structures the graph benchmarks run on. |
| [`hnsw/`](hnsw/) | Oblivious HNSW, built on OSAM+ and BOSAM. `extract_hnsw.cpp` exports a FAISS index to JSON, `o_hnsw.py` partitions the hnsw and answers queries obliviously, and `run.sh` drives both. `plots/` turns run logs into the paper's figures and tables. |
| [`replay_rust/`](replay_rust/) | Optional native PathORAM stash replay (PyO3). Without it the Python replay is used automatically. |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

- **METIS partitioning** needs `libmetis` on the system for `pymetis` to build.
  Everything else runs without it; `o_hnsw.py` reports the missing dependency
  rather than silently changing the layout.
- **Native stash replay** (optional, faster `--track-stash` runs):
  ```bash
  pip install maturin && maturin develop --release --manifest-path replay_rust/Cargo.toml
  ```
- **`extract_hnsw`** is only needed to convert a *new* FAISS index; the LAION-1k
  and 10k JSON graphs are already in the repo. Build instructions:
  [Converting a FAISS HNSW Index to JSON](hnsw/README_extract_hnsw.md).

## Quick start

### Oblivious HNSW

```bash
cd hnsw
./run.sh --partitioning metis --metric ip          # LAION-1k, 100 queries, METIS
./run.sh data/1k/hnsw_32_40.index 100 \
    --partitioning metis --metric ip --ef 64 --k 10 --bs 4096 \
    --pointer-type multiwriterary                  # same index under BOSAM

# 10k needs its JSON graph named explicitly -- see the note below
./run.sh data/10k/hnsw_32_40.index 100 --json data/10k/hnsw_32_40.json \
    --partitioning metis --metric ip
```

`run.sh` falls back to `data/1k/hnsw_32_40.json` whenever `--json` is omitted,
whatever index is named on the command line. Pass `--json` (or
`--generate-json`) for any dataset other than 1k, or the run will search the
1k graph.

`./run.sh --help` lists every option. The ones specific to this work:

| Option | Default | Notes |
| --- | --- | --- |
| `--partitioning METHOD` | `none` | `none`, `metis`, or `kmeans`. Partitions the base layer (L0). |
| `--n-clusters N` | `sqrt(ntotal)` | Number of partitions, for `metis` and `kmeans`. |
| `--pointer-type TYPE` | `multiwrite` | Pointer scheme; see the table below. |
| `--bs N` | `64` | Block size in bytes. The `multiwriterary` branching factor is derived from it as `floor((bs - 16) / 8)` unless `--pointer-branching-factor` overrides it. Inert for the other pointer types. |
| `--track-stash` | off | Replays the PathORAM trace and reports build- and query-phase stash occupancy. Requires `--path-oram-tree-size`. |

`metis` gives tightly balanced partitions; `kmeans` groups by vector proximity
and its partition sizes vary widely.

### Oblivious graph algorithms

```bash
python python-implementation/benchmark_single_alg.py \
    --n 1024 --d 20 --bs 64 --trials 5 \
    --pt multiwrite --alg rw --wl 50 \
    --move --no_copies --prime --staticinsertion
```

Builds an Erdős–Rényi graph on `--n` nodes with average degree `--d`, then runs
`--alg` for `--trials` trials and reports accesses broken down by structure.

`--alg` accepts `rw` (random walk), `cd` (contact discovery), `pr` (PageRank),
`dtc` (directed triangle count), `bfs`, `dfs`, `dijkstra`, and `prim`. Real
edge lists go through `benchmark_graph_from_csv.py` with the same flags plus
`--graph_name`, `--file`, and `--directed` / `--undirected`.

## Pointer schemes

`--pointer-type` (HNSW) and `--pt` (graph benchmarks) select the representation
of incoming pointers. This is the only difference between the two schemes the
paper compares — the graph, the partitions, and the queries are identical.

| Flag value | Paper | Underlying SAM | Structure |
| --- | --- | --- | --- |
| `multiwrite` | OSAM+ | multi-write / single-read | binary incoming-pointer tree (default) |
| `multiwriterary` | BOSAM | multi-write / single-read | r-ary (blocked) pointer tree, fan-out `r` — **this work** |
| `original` | — | single-write / single-read | baseline pointer, from OSAM+ |
| `recursive` | — | multi-write / multi-read | recursive pointer, from OSAM+ |

## Reproducing the experiments

### HNSW block-size sweep

Produces the latency-vs-block-size figure. The block size is not chosen freely:
it is *derived* from the partition layout, so a whole partition fits in one ORAM
block and `get_neighbors_in_cluster` can fetch it in a single read.

```bash
cd hnsw

# 1. Partition at each count, measure the largest partition's payload, round up
#    to a power of two. Minutes at most; no OSAM work.
python probe_partitions.py --graph data/10k/hnsw_32_40.json --dataset 10k \
    --n-clusters 2 4 8 16 32 63 125 250 500 1000

# 2. Run both pointer types at every block size the probe marked run=1.
./exp_blocksize_sweep.sh --dataset 10k

# 3. Figures, at each link speed the paper reports.
python plots/plot_latency.py --logs logs/10k_bssweep --dataset 10k \
    --bandwidth 100e6 --out plots/out_bw100MBps/latency_10k_100
python plots/plot_latency.py --logs logs/10k_bssweep --dataset 10k \
    --bandwidth 500e6 --out plots/latency_10k_500
```

`--bandwidth` is in bytes/second, so `100e6` and `500e6` are the 100 MB/s and
500 MB/s links; `--rtt` (default `0.1`, seconds).

`--dataset` accepts `1k`, `10k`, `sift1m`, `laion1m`, and `msmarco1m`. 

<!-- **Sweep the partition count, not `--bs` directly.** `--bs` only reaches the
r-ary branching factor, so it is inert for `multiwrite`; a fixed-partition sweep
leaves OSAM+ at a flat read count no matter the block size. Varying
`--n-clusters` changes partition granularity, which both schemes see, and the
block size follows from it. The size model is the plaintext one,
`4 * (n_nodes + intra_edges + 2 * cross_edges)` bytes per partition, matching the
`HNSWCluster` payload — node ids, intra adjacency, and cross edges, but no
vectors — so the probe's predicted partition sizes match the run's own figures. -->


### Figures and tables

The plotting scripts read the raw `o_hnsw.py` stdout logs under `hnsw/logs/` —
there are no intermediate data files to keep in sync.

```bash
cd hnsw
python plots/plot_latency.py   --logs logs/<run>   # end-to-end latency vs. block size
python plots/make_tables.py                        # paper tables -> paper/tables/*.tex
```

## Datasets

| Dataset | N | dim | M | efC | metric | How to obtain |
| --- | --- | --- | --- | --- | --- | --- |
| LAION-1k / 10k | 1e3 / 1e4 | 512 | 32 | 40 | IP | ships with the repository |
| LAION-1M | 1,000,448 | 512 | 32 | 40 | IP | `download_data.sh` |
| SIFT-1M | 1,000,000 | 128 | 64 | 80 | L2 | `download_data.sh` |
| TripClick | 1,523,871 | 768 | 128 | 160 | IP | `download_data.sh` |
| MSMARCO | 1,000,000 | 768 | 128 | 200 | IP | `download_data.sh` |

**LAION-1k and LAION-10k ship with the repository** — indices, JSON graphs, base
and query vectors, and ground truth — so the quick start and the 1k/10k sweeps
run with no downloads.

The larger datasets come from the `compass_osdi` GCS bucket.
`data/download_data.sh` takes no arguments and downloads **all** of them in one
pass: it lists every key under `dataset/` — 74 objects across `sift`,
`laion1m`, `trip_distilbert`, `msmarco_bert`, and `tfidf` — and writes them to
`./data/dataset/<name>/`, relative to the directory it is run from.

```bash
cd hnsw/data && ./download_data.sh
```

To fetch a single dataset instead, narrow `PREFIX` at the top of the script,
e.g. `PREFIX="dataset/sift/"`.

The bucket's layout is not the layout the sweep reads, so move each corpus into
place afterwards:

| `--dataset` | Expected location, relative to `hnsw/` |
| --- | --- |
| `sift1m` | `data/sift/` (`base.fvecs`, `query.fvecs`) and `data/sift/data/dataset/sift/` (index, `gt.ivecs`) |
| `laion1m` | `data/laion/1m/` |
| `tripclick` | `data/tripclick/data/dataset/trip_distilbert/` |
| `msmarco1m` | `data/msmarco/data/dataset/msmarco_bert/` |


### Converting an hnsw index to JSON

`o_hnsw.py` reads a JSON graph, not a FAISS index. Any newly built index needs
one export:

```bash
./build/extract_hnsw <index> <out.json>     # build-linux/extract_hnsw on Linux
```

The `run_tripclick_blocksize.sh` and `run_msmarco_blocksize.sh` wrappers do this
step themselves.


## Documentation

- [Converting a FAISS HNSW Index to JSON](hnsw/README_extract_hnsw.md)
- [Plaintext HNSW Partitioning (METIS / K-means)](hnsw/cluster/plaintext/README_partitioning.md)
