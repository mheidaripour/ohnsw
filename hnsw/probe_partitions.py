#!/usr/bin/env python3
"""Predict the ORAM block size a METIS partition count will need.

The block-size sweep works backwards: pick a partition count, partition the
base layer, measure how many bytes the largest partition's payload occupies,
and round that up to a power of two. That rounded value is what gets passed to
``o_hnsw.py --bs``, so a whole partition fits in one block and
``get_neighbors_in_cluster`` can fetch it in a single ORAM read.

Running the real thing just to learn the partition sizes would cost hours, so
this probe reproduces only the cheap prefix: parse the graph JSON, symmetrize
the base layer, call METIS. It never touches the OSAM machinery.

Every step is deliberately byte-identical to the corresponding step in
``o_hnsw.py`` -- see ``build_symmetric_adjacency`` and ``resolve_n_parts`` --
because a probe that partitions differently from the run predicts a block size
for a layout that never materializes. ``verify_probe.py`` checks the agreement
against logs of runs that already happened.

Usage:
    python probe_partitions.py --graph data/1k/hnsw_32_40.json --dataset 1k \\
        --n-clusters 2 4 8 16 31 62 125 250
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

try:
    import pymetis
    _METIS_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    _METIS_AVAILABLE = False


HERE = Path(__file__).resolve().parent

# Size model lifted from cluster/plaintext/hnsw-metis-partitioned.py, which
# charges 4 bytes per stored integer:
#     graph_bytes = 4 * (n_nodes + intra_edges + 2 * cross_edges)
# One int per node id, one per intra adjacency entry, two per cross edge
# (neighbour id + its cluster id). That is exactly the HNSWCluster payload in
# o_hnsw.py -- `nodes`, `neighbors`, `cross` -- and deliberately excludes
# vectors, which o_hnsw keeps in plaintext outside the ORAM.
BYTES_PER_INT = 4

# o_hnsw.py:2127 derives the r-ary fan-out as floor((bs - 2*8) / 8): 8-byte
# slots behind a 16-byte header. The `nodes` and `copies` models below invert
# that, so they answer "how big must a block be for one deref to reach N
# slots".
SLOT_BYTES = 8
HEADER_BYTES = 16

CSV_FIELDS = [
    "dataset",
    "graph",
    "graph_size",
    "graph_mtime",
    "ntotal",
    "n_clusters_requested",
    "num_partitions",
    "max_partition_nodes",
    "avg_partition_nodes",
    "min_partition_nodes",
    "intra_edges",
    "cross_edges",
    "max_incoming_cross",
    "edge_cuts",
    "max_graph_bytes",
    "avg_graph_bytes",
    "bs_bytes",
    "bs_nodes",
    "bs_copies",
    "block_size_bytes",
    "clamped",
    "run",
]

MODELS = ("bytes", "nodes", "copies")


def next_pow2(n: int) -> int:
    """Smallest power of two >= n (and >= 1)."""
    if n <= 1:
        return 1
    return 1 << (int(n) - 1).bit_length()


def load_base_layer(json_path: str | Path):
    """Parse the graph JSON into ``(ntotal, {node_id: [neighbour, ...]})``.

    Mirrors o_hnsw.py:1486-1500 for level 0 only: neighbours are taken verbatim
    from the JSON, and a node with no level-0 entry gets an empty list rather
    than being absent.

    Only the base layer is kept. o_hnsw's own loader also materializes an edge
    list over every level -- ~84M tuples on sift1m -- that partitioning never
    reads.
    """
    with open(json_path, "r") as f:
        j = json.load(f)

    ntotal = int(j["meta"]["ntotal"])
    base_layer = {i: [] for i in range(ntotal)}

    for node_obj in j["graph"]:
        node_id = int(node_obj["node"])
        if node_id >= ntotal:
            continue
        for level_obj in node_obj["levels"]:
            if int(level_obj["layer"]) == 0:
                base_layer[node_id] = [int(x) for x in level_obj["neighbors"]]
                break

    return ntotal, base_layer


def build_symmetric_adjacency(base_layer: dict, ntotal: int):
    """Undirected adjacency lists for METIS.

    Copied from o_hnsw.py:646-657 and must stay identical to it -- same
    self-loop skip, same ``>= ntotal`` guards on both endpoints, same
    ``sorted()``. METIS output depends on the exact adjacency it is handed, so
    any divergence here silently makes the probe describe a different
    partitioning than the run produces.
    """
    adjacency_sets = [set() for _ in range(ntotal)]
    for node_id, neighbors in base_layer.items():
        if node_id >= ntotal:
            continue
        for neighbor in neighbors:
            if neighbor != node_id and neighbor < ntotal:
                adjacency_sets[node_id].add(neighbor)
                adjacency_sets[neighbor].add(node_id)

    return [sorted(neighbors) for neighbors in adjacency_sets]


def resolve_n_parts(n_clusters: int, ntotal: int) -> int:
    """The partition count the run will actually ask METIS for.

    ``_default_n_clusters`` clamp (o_hnsw.py:531-535) followed by the
    too-small-graph adjustment (o_hnsw.py:662-664). Requesting 250 partitions
    of a 1k graph, for instance, silently becomes 100.
    """
    n_parts = max(2, min(int(n_clusters), ntotal))
    if ntotal < n_parts * 2:
        n_parts = max(2, ntotal // 10)
    return n_parts


def partition(adjacency_list, ntotal: int, n_parts: int):
    """Run METIS and re-index partition ids contiguously.

    Returns ``(node_to_part, num_partitions, edge_cuts)``. The re-indexing
    matches o_hnsw.py:678-686: METIS can leave gaps, and empty partitions are
    dropped, so the realized count can come in under ``n_parts``.
    """
    edge_cuts, partition_vector = pymetis.part_graph(n_parts, adjacency=adjacency_list)

    raw = {}
    for node_id, part_id in enumerate(partition_vector):
        raw.setdefault(int(part_id), []).append(node_id)

    node_to_part = {}
    num_partitions = 0
    for new_id, nodes in enumerate(v for v in raw.values() if v):
        for node_id in nodes:
            node_to_part[node_id] = new_id
        num_partitions = new_id + 1

    return node_to_part, num_partitions, edge_cuts


def partition_stats(base_layer: dict, node_to_part: dict, num_partitions: int):
    """Per-partition node/edge counts and payload sizes.

    Edges are counted over the *directed*, unfiltered level-0 adjacency, which
    is what ``_build_base_layer_graph`` hands to
    ``_build_cluster_graph_structure`` (o_hnsw.py:514-529, 716-726). Note this
    differs from the METIS adjacency above, which is symmetrized and filtered
    -- the run uses one for partitioning and the other for accounting, and the
    probe has to match both.

    ``max_incoming_cross`` counts cross edges arriving at each partition. The
    run takes one ``SmartPointer.copy`` per cross edge, so this drives the
    ``Max copies / cluster`` figure and, with it, the r-ary fan-out the pointer
    tree needs to stay flat.
    """
    n_nodes = [0] * num_partitions
    intra = [0] * num_partitions
    cross_out = [0] * num_partitions
    cross_in = [0] * num_partitions

    for node_id, part_id in node_to_part.items():
        n_nodes[part_id] += 1

    for src, neighbors in base_layer.items():
        src_part = node_to_part.get(src)
        if src_part is None:
            continue
        for dst in neighbors:
            dst_part = node_to_part.get(dst)
            if dst_part is None:
                continue
            if src_part == dst_part:
                intra[src_part] += 1
            else:
                cross_out[src_part] += 1
                cross_in[dst_part] += 1

    graph_bytes = [
        BYTES_PER_INT * (n_nodes[p] + intra[p] + 2 * cross_out[p])
        for p in range(num_partitions)
    ]

    return {
        "max_partition_nodes": max(n_nodes),
        "min_partition_nodes": min(n_nodes),
        "avg_partition_nodes": sum(n_nodes) / num_partitions,
        "intra_edges": sum(intra),
        "cross_edges": sum(cross_out),
        # The run reports max(cluster_counts) - 1, where cluster_counts starts
        # at 1 for the partition's own entry pointer (o_hnsw.py:1897). Match
        # that definition so the column is directly comparable to a log.
        "max_incoming_cross": max(cross_in),
        "max_graph_bytes": max(graph_bytes),
        "avg_graph_bytes": sum(graph_bytes) / num_partitions,
    }


def block_sizes(stats: dict) -> dict:
    """The three candidate block-size models, before clamping.

    ``bytes`` is the default and the one the sweep uses: the measured payload
    of the largest partition, rounded up. The other two answer a different
    question -- how wide the r-ary pointer tree must be -- and are carried
    along because the fan-out threshold they describe is what makes r-ary's
    reads/query fall off a cliff at a particular block size.
    """
    return {
        "bs_bytes": next_pow2(stats["max_graph_bytes"]),
        "bs_nodes": next_pow2(SLOT_BYTES * stats["max_partition_nodes"] + HEADER_BYTES),
        "bs_copies": next_pow2(
            SLOT_BYTES * (stats["max_incoming_cross"] + 1) + HEADER_BYTES
        ),
    }


def probe_one(adjacency_list, base_layer, ntotal, n_clusters, model, min_bs, max_bs):
    """One CSV row: partition at ``n_clusters``, measure, derive a block size."""
    n_parts = resolve_n_parts(n_clusters, ntotal)
    node_to_part, num_partitions, edge_cuts = partition(adjacency_list, ntotal, n_parts)

    stats = partition_stats(base_layer, node_to_part, num_partitions)
    sizes = block_sizes(stats)

    raw_bs = sizes[f"bs_{model}"]
    bs = min(max(raw_bs, min_bs), max_bs)

    row = {
        "n_clusters_requested": n_clusters,
        "num_partitions": num_partitions,
        "edge_cuts": edge_cuts,
        "block_size_bytes": bs,
        "clamped": int(bs != raw_bs),
        "run": 1,
    }
    row.update(stats)
    row.update(sizes)
    row["avg_partition_nodes"] = round(row["avg_partition_nodes"], 1)
    row["avg_graph_bytes"] = round(row["avg_graph_bytes"], 1)
    return row


def mark_runnable(rows):
    """Drop duplicate block sizes, keeping the smallest partition count.

    Adjacent partition counts can round to the same power of two. Two runs at
    one x-position is wasted compute, and worse, ``dedupe_by(key=block_size)``
    in the plot scripts would quietly discard one of them by mtime. Resolving
    it here makes the choice explicit and visible in the CSV.

    The smallest n_clusters wins because it has the largest partitions, hence
    the fewest reads per query -- the better point at that block size.
    """
    best = {}
    for row in rows:
        bs = row["block_size_bytes"]
        if bs not in best or row["n_clusters_requested"] < best[bs]["n_clusters_requested"]:
            best[bs] = row
    for row in rows:
        row["run"] = int(row is best[row["block_size_bytes"]])
    return rows


def read_cache(out_path: Path, graph_meta: dict):
    """Rows from a previous probe of the same graph, keyed by n_clusters.

    Guarded on path, size and mtime: a regenerated graph invalidates every
    cached row, and silently reusing them would pin the sweep to block sizes
    derived from data that no longer exists.
    """
    if not out_path.is_file():
        return {}

    try:
        with open(out_path, newline="") as f:
            rows = list(csv.DictReader(f))
    except (OSError, csv.Error) as e:
        print(f"  ! ignoring unreadable cache {out_path}: {e}", file=sys.stderr)
        return {}

    cached = {}
    for row in rows:
        same_graph = (
            row.get("graph") == graph_meta["graph"]
            and row.get("graph_size") == str(graph_meta["graph_size"])
            and row.get("graph_mtime") == str(graph_meta["graph_mtime"])
            and row.get("ntotal") == str(graph_meta["ntotal"])
        )
        if not same_graph:
            continue
        try:
            cached[int(row["n_clusters_requested"])] = {
                k: (int(v) if v.lstrip("-").isdigit() else v) for k, v in row.items()
            }
        except (KeyError, ValueError):
            continue
    return cached


def write_csv(out_path: Path, rows):
    """Atomic write, so an interrupted probe never leaves a half CSV behind.

    ``lineterminator="\\n"`` is not cosmetic: csv defaults to CRLF, which
    leaves a trailing ``\\r`` on the last column of every line. The shell
    driver looks columns up by header name, and ``run\\r`` does not match
    ``run``, so the whole sweep reads as "no configs to run".
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=CSV_FIELDS, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, out_path)


def fmt_bytes(n: float) -> str:
    for unit, scale in (("B", 1), ("KiB", 2 ** 10), ("MiB", 2 ** 20), ("GiB", 2 ** 30)):
        if n < scale * 1024:
            return f"{n / scale:g} {unit}"
    return f"{n / 2 ** 40:g} TiB"


def print_table(rows):
    head = (
        f"{'n_clust':>8} {'parts':>7} {'max nodes':>10} {'max bytes':>12} "
        f"{'block size':>11} {'bs_nodes':>10} {'bs_copies':>11} {'run':>4}"
    )
    print(f"\n{head}")
    print("-" * len(head))
    for row in rows:
        print(
            f"{row['n_clusters_requested']:>8} {row['num_partitions']:>7} "
            f"{row['max_partition_nodes']:>10,} {row['max_graph_bytes']:>12,} "
            f"{fmt_bytes(row['block_size_bytes']):>11} "
            f"{fmt_bytes(row['bs_nodes']):>10} {fmt_bytes(row['bs_copies']):>11} "
            f"{row['run']:>4}"
        )

    skipped = [r for r in rows if not r["run"]]
    if skipped:
        print(
            f"\n  {len(skipped)} config(s) marked run=0 -- duplicate block size: "
            + ", ".join(f"nc={r['n_clusters_requested']}" for r in skipped)
        )
    clamped = [r for r in rows if r["clamped"]]
    if clamped:
        print(
            f"  {len(clamped)} config(s) clamped to the --min-bs/--max-bs range: "
            + ", ".join(f"nc={r['n_clusters_requested']}" for r in clamped)
        )


def main():
    p = argparse.ArgumentParser(
        description="Predict per-partition-count ORAM block sizes for the sweep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--graph", required=True, help="HNSW graph JSON")
    p.add_argument("--dataset", default=None,
                   help="short name for the CSV and default output path "
                        "(default: the graph's parent directory name)")
    p.add_argument("--n-clusters", type=int, nargs="+", required=True,
                   help="partition counts to probe")
    p.add_argument("--out", default=None,
                   help="output CSV (default: sweep/partitions_<dataset>.csv)")
    p.add_argument("--model", choices=MODELS, default="bytes",
                   help="which derivation feeds block_size_bytes")
    p.add_argument("--min-bs", type=int, default=4096, help="block size floor, bytes")
    p.add_argument("--max-bs", type=int, default=1 << 22, help="block size ceiling, bytes")
    p.add_argument("--force", action="store_true", help="recompute cached rows")
    p.add_argument("--quiet", action="store_true", help="suppress the table")
    args = p.parse_args()

    if not _METIS_AVAILABLE:
        raise SystemExit(
            "pymetis is required. Install with: pip install pymetis (needs libmetis)."
        )

    graph_path = Path(args.graph)
    if not graph_path.is_file():
        raise SystemExit(f"graph not found: {graph_path}")

    dataset = args.dataset or graph_path.parent.name
    out_path = Path(args.out) if args.out else HERE / "sweep" / f"partitions_{dataset}.csv"

    stat = graph_path.stat()
    graph_meta = {
        "dataset": dataset,
        "graph": str(graph_path),
        "graph_size": stat.st_size,
        "graph_mtime": int(stat.st_mtime),
        "ntotal": None,  # filled in after the parse
    }

    wanted = sorted(set(args.n_clusters))

    print(f"Loading {graph_path} ...")
    t0 = time.perf_counter()
    ntotal, base_layer = load_base_layer(graph_path)
    graph_meta["ntotal"] = ntotal
    print(f"  {ntotal:,} nodes, {time.perf_counter() - t0:.1f}s")

    cached = {} if args.force else read_cache(out_path, graph_meta)
    todo = [nc for nc in wanted if nc not in cached]
    if cached:
        reused = [nc for nc in wanted if nc in cached]
        print(f"  reusing {len(reused)} cached row(s) from {out_path}")

    rows = []
    if todo:
        t0 = time.perf_counter()
        adjacency_list = build_symmetric_adjacency(base_layer, ntotal)
        undirected = sum(len(a) for a in adjacency_list) // 2
        print(f"  symmetrized: {undirected:,} undirected edges, "
              f"{time.perf_counter() - t0:.1f}s")

        for nc in todo:
            t0 = time.perf_counter()
            row = probe_one(
                adjacency_list, base_layer, ntotal, nc,
                args.model, args.min_bs, args.max_bs,
            )
            row.update(graph_meta)
            rows.append(row)
            print(f"  nc={nc:<7} -> {row['num_partitions']:,} parts, "
                  f"max {row['max_partition_nodes']:,} nodes / "
                  f"{row['max_graph_bytes']:,} B -> "
                  f"{fmt_bytes(row['block_size_bytes'])}  "
                  f"({time.perf_counter() - t0:.1f}s)")

    rows.extend(cached[nc] for nc in wanted if nc in cached)
    rows.sort(key=lambda r: r["n_clusters_requested"])
    mark_runnable(rows)

    write_csv(out_path, rows)
    if not args.quiet:
        print_table(rows)
    print(f"\n✓ wrote {len(rows)} row(s) to {out_path}")
    print(f"  {sum(r['run'] for r in rows)} marked for running "
          f"({2 * sum(r['run'] for r in rows)} o_hnsw.py runs)")


if __name__ == "__main__":
    main()
