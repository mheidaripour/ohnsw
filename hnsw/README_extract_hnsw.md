# Converting a FAISS HNSW Index to JSON

`extract_hnsw` reads a FAISS HNSW index file and exports its graph
structure (nodes, per-layer neighbor lists, degree stats) as JSON.

## Prerequisites

- `cmake` >= 3.15, a C++17 compiler (`g++`/`clang++`)
- BLAS (OpenBLAS or similar). On Ubuntu, if `libopenblas-dev` isn't
  installable (e.g. not packaged for your release), you can instead
  symlink the runtime lib so CMake can find it without root:
  ```bash
  mkdir -p ~/.local/lib
  ln -sf /lib/x86_64-linux-gnu/libopenblas.so.0 ~/.local/lib/libopenblas.so
  ```

## Step 1: Set up FAISS (via conda)

You need a FAISS **C++** install (headers + `libfaiss.so`) — the
Python-only `pip install faiss-cpu` wheel does **not** ship the
headers needed here. The easiest sudo-free option is a conda env:

```bash
conda create -n ann_osam -c conda-forge faiss-cpu=1.12.0 python=3.12
conda activate ann_osam
pip install -r ./osam/requirements.txt
```

This installs FAISS's headers, `libfaiss.so`, and a CMake package
config under the env's prefix (e.g. `$CONDA_PREFIX/include/faiss`,
`$CONDA_PREFIX/lib/libfaiss.so`).

## Step 2: Build with CMake

From `osam/hnsw/`:

```bash
export FAISS_ROOT=$CONDA_PREFIX          # points CMake at the conda env's FAISS
export CMAKE_PREFIX_PATH=$HOME/.local    # for the OpenBLAS symlink

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

This produces `build/extract_hnsw` (and `build/search_hnsw`, which
additionally needs `nlohmann/json.hpp` — install via
`conda install -c conda-forge nlohmann_json` if you need that target too).

The resulting binary bakes in the FAISS lib path via `RUNPATH`, so it
runs standalone — no need to keep the conda env activated or set
`LD_LIBRARY_PATH` at run time.

## Step 3: Run

```bash
./build/extract_hnsw <faiss_index_file> <output.json>
```

Example:

```bash
./build/extract_hnsw \
  data/sift/data/dataset/sift/hnsw_64_80_2_ip.index \
  data/sift/hnsw_64_80_2_ip.json
```

Output:
- Prints per-layer stats to stdout (`nodes`, `edges`, `avg/min/max/p90 degree`).
- Writes a JSON file containing `meta` (ntotal, d, metric, entry_point,
  max_level, efConstruction/efSearch), raw `tables` (levels/offsets/
  neighbors), `layer_stats`, and the full per-node `graph`.

The JSON produced here is the input to the partitioning experiments in
[cluster/plaintext/README_partitioning.md](cluster/plaintext/README_partitioning.md).

## Troubleshooting

- **`CMake Error ... Target "FAISS::faiss" not found`**: FAISS's own
  CMake export names the imported target plain `faiss`, not
  `FAISS::faiss`. Already fixed in `CMakeLists.txt`.
- **`fatal error: nlohmann/json.hpp: No such file or directory`**:
  only affects `search_hnsw`, not `extract_hnsw`. Install
  `nlohmann_json` via conda or apt (`nlohmann-json3-dev`), or ignore
  if you only need `extract_hnsw`.
- **FAISS not found while configuring**: make sure `FAISS_ROOT` points
  at a prefix containing `include/faiss/Index.h` and `lib/libfaiss.so`.
