#!/usr/bin/env python3
"""Predict the ORAM block size a NON-partitioned oblivious HNSW run needs.

``probe_partitions.py`` answers this for ``--partitioning metis`` by measuring the
largest partition's payload. That model does not carry over here, because without
partitioning **nothing stored grows with n**:

    Vertex                  exactly --branching-factor pointers (default 2)
    FanOut                  branching_factor + 1
    AVLNode                 exactly 2, hard-coded
    multiwriterary NODE     exactly bf slots -- fits --bs by construction
    multiwrite INNER        exactly 2, hard-coded -> --bs is inert

so a "largest payload" model degenerates to a constant at every scale.

The one quantity that does grow is the number of ``SmartPointer.copy`` calls per
pointee. In the non-clustered build (oblivious_graph.py:1191-1198) that count is
the destination's **in-degree**. It reaches the pointer as
``copy_raw_address(old_head, num_copies)``, whose Case 1 -- the entry pointer is
still a fresh ROOT cell -- sets ``total_heads = num_copies + 1`` and calls
``_build_subtree(new_root, total_heads)`` (smart_pointer_multi_write_rary.py:542-555).
``_build_subtree`` (ibid:1021-1036) takes its flat branch if and only if
``n <= bf``, and o_hnsw.py:2128 derives ``bf = floor((bs - 16) / 8)``. So a
pointee's reference tree is depth-1 exactly when::

    bs >= 8 * (in_degree + 1) + 16

and the block size the run needs is that, over the worst pointee, rounded up::

    bs_copies = next_pow2(8 * (max_in_degree + 1) + 16)

Following every sweep already in this repo, the statistic is a **max**, not a
percentile: sweep/partitions_*.csv were all produced with ``--model bytes``, whose
``max_graph_bytes`` is a max over partitions. ``--percentile`` can trade that away,
and the ``pointees_over_bf`` / ``frac_derefs_over_bf`` / ``avg_ref_depth`` columns
price the trade per dataset rather than asserting it.

Two things this probe deliberately does differently from probe_partitions.py:

* **It reads every level, not just level 0.** o_hnsw.py:1502-1512 emits
  ``(u, v, level)`` for every level and ``emulate_graph`` collapses them through
  ``DG.add_weighted_edges_from`` (oblivious_graph.py:1071), so duplicate ``(u,v)``
  pairs across levels dedup to one edge. ``load_base_layer``'s level-0-only view is
  correct for partitioning and wrong here -- 43.36M vs 43.05M edges on laion1M.
* **It keys rows on --bs, not on a partition count.** There is no partition axis;
  every row is a candidate block size, and all of them are runnable.

Caveat worth stating plainly: the SAM stores one arbitrary Python object per
address (single_access_machine.py:25). There is no serializer and no size check
anywhere in the implementation, so "block size" is a modeling convention applied at
the CLI boundary and in the plots. The copies model is the only one with mechanical
consequences inside a run, because ``bf`` genuinely changes the tree
``_build_subtree`` builds. ``bs_bytes`` and ``bs_nodes`` are reported so the reader
can see why they drop out, and are flagged non-binding / advisory.

Related: the partitioned sweep's ``bytes`` model sometimes undershoots its own
``copies`` constraint -- partitions_tripclick.csv at nc=3000 chose bs_bytes=1 MiB
while bs_copies=4 MiB, so those reference trees were not depth-1.

WHAT THE RECOMMENDATION DOES AND DOES NOT BUY (measured, laion 1k, 20 queries,
--partitioning none --pointer-type multiwriterary --branching-factor 2):

    --bs      reads/query   writes/query   R@10
      64         14524.2        14903.0   1.0000
     256         14485.4        15666.9   1.0000
    1024         14462.5        18314.5   1.0000
    4096         14461.5        18946.2   1.0000   <- bs_copies
    8192         14461.5        18946.2   1.0000

The prediction is exact in the place it claims: reads/query fall monotonically and
go *identical* at 4096 and 8192, which is where ``bs_copies`` said the last
reference tree goes flat. But the effect is small -- 0.4% -- because without
partitioning the per-query cost is dominated by the out-tree walk, not the pointer
deref. With ``--branching-factor 2`` a 43-neighbour vertex is a depth-6 binary
out-tree, which is where ~15k reads/query come from; the reference tree contributes
1-3.

Writes move the *opposite* way, and by much more: +27% from bs=64 to bs=4096. A
wider group means ``normalize_group`` rewrites more members per splay, so search
writes grow with bf even as reads shrink. ``bs_copies`` minimizes reads; it does not
minimize total memory traffic. If writes are the metric of interest, the cheapest
point is the *smallest* block, not the recommended one -- which is a genuinely
different answer from the partitioned case, where a block that cannot hold a
partition is simply wrong.

Negative control, same config with ``--pointer-type multiwrite`` at bs=64 and
bs=4096: 14529.0 reads/query, 14667.5 writes/query, 293,350 total writes, R@10
1.0000 at both. Byte-identical, confirming ``--bs`` is inert for multiwrite here.

Usage:
    python probe_nopart.py --graph data/1k/hnsw_32_40.json --dataset 1k
    python probe_nopart.py --graph data/1k/hnsw_32_40.json --dataset 1k --verify-build
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import sys
import time
from array import array
from pathlib import Path

from probe_partitions import (
    next_pow2,
    fmt_bytes,
    BYTES_PER_INT,
    SLOT_BYTES,
    HEADER_BYTES,
)

HERE = Path(__file__).resolve().parent

CSV_FIELDS = [
    "dataset",
    "graph",
    "graph_size",
    "graph_mtime",
    "ntotal",
    "graph_nodes",
    "distinct_edges",
    "self_loops",
    "max_level",
    "max_in_degree",
    "p999_in_degree",
    "p99_in_degree",
    "mean_in_degree",
    "max_out_degree",
    "p99_out_degree",
    "mean_out_degree",
    "branching_factor",
    "bf_out_needed",
    "bs_bytes",
    "bs_nodes",
    "bs_copies",
    "bs_copies_p99",
    "block_size_bytes",
    "bf_at_bs",
    "pointees_over_bf",
    "derefs_over_bf",
    "frac_derefs_over_bf",
    "avg_ref_depth",
    "build_ref_cells",
    "binding",
    "recommended",
    "clamped",
    "run",
]

PERCENTILES = ("100", "99.9", "99")


def bf_for_bs(bs: int) -> int:
    """The r-ary fan-out o_hnsw.py:2128 derives from a block size."""
    return (bs - HEADER_BYTES) // SLOT_BYTES


def bs_for_slots(slots: int) -> int:
    """Smallest power-of-two block whose fan-out reaches ``slots`` pointers."""
    return next_pow2(SLOT_BYTES * slots + HEADER_BYTES)


def load_all_levels(json_path: str | Path):
    """Parse the graph JSON into the DiGraph o_hnsw actually builds.

    Mirrors ``_load_hnsw_json_to_graph`` (o_hnsw.py:1467-1520) followed by
    ``DG.add_weighted_edges_from`` (oblivious_graph.py:1071): every level
    contributes edges, and duplicate ``(u, v)`` pairs across levels collapse to
    one. Unlike probe_partitions.load_base_layer this is not level-0 only, and it
    does not pre-seed absent nodes -- a node with no edge at any level is not in
    the DiGraph, so ``add_vertex`` never runs for it and it owns no pointer.

    Returns degree counters rather than an edge list: 43.3M ``(u, v)`` tuples
    would cost several GB, and nothing downstream needs the edges themselves.

    One deliberate divergence from load_base_layer, which silently skips ids
    ``>= ntotal``: o_hnsw would raise IndexError on them, so they are counted and
    warned about instead. A graph o_hnsw cannot load should not receive a
    confident block-size prediction.
    """
    with open(json_path, "r") as f:
        j = json.load(f)

    meta = j["meta"]
    ntotal = int(meta["ntotal"])
    max_level = int(meta["max_level"])

    in_deg = array("i", bytes(4 * ntotal))
    out_deg = array("i", bytes(4 * ntotal))
    present = bytearray(ntotal)

    distinct_edges = 0
    self_loops = 0
    out_of_range = 0

    for node_obj in j["graph"]:
        node_id = int(node_obj["node"])
        if node_id >= ntotal:
            out_of_range += 1
            continue

        neighbors = set()
        for level_obj in node_obj["levels"]:
            neighbors.update(int(x) for x in level_obj["neighbors"])

        if not neighbors:
            continue

        present[node_id] = 1
        out_deg[node_id] = len(neighbors)
        distinct_edges += len(neighbors)

        for dst in neighbors:
            if dst == node_id:
                self_loops += 1
            if dst >= ntotal:
                out_of_range += 1
                continue
            present[dst] = 1
            in_deg[dst] += 1

    del j

    if out_of_range:
        print(
            f"  ! {out_of_range} node/neighbour id(s) >= ntotal; o_hnsw.py would "
            f"raise IndexError loading this graph",
            file=sys.stderr,
        )

    return {
        "ntotal": ntotal,
        "max_level": max_level,
        "graph_nodes": sum(present),
        "distinct_edges": distinct_edges,
        "self_loops": self_loops,
        "in_deg": in_deg,
        "out_deg": out_deg,
        "present": present,
    }


def degree_stats(deg, present, label):
    """max / mean / percentiles over nodes that are actually in the DiGraph."""
    vals = sorted(d for d, p in zip(deg, present) if p)
    if not vals:
        return {
            f"max_{label}": 0,
            f"mean_{label}": 0.0,
            f"p50_{label}": 0,
            f"p99_{label}": 0,
            f"p999_{label}": 0,
        }

    n = len(vals)

    def pct(q):
        return vals[min(n - 1, int(q / 100.0 * n))]

    return {
        f"max_{label}": vals[-1],
        f"mean_{label}": sum(vals) / n,
        f"p50_{label}": pct(50),
        f"p99_{label}": pct(99),
        f"p999_{label}": pct(99.9),
    }


_DEPTH_MEMO: dict = {}
_CELLS_MEMO: dict = {}


def subtree_leaf_depths(n: int, bf: int) -> dict:
    """Leaf depths of ``_build_subtree(parent, n)``, as ``{depth: count}``.

    An exact mirror of smart_pointer_multi_write_rary.py:1021-1036, including the
    ``divmod(n, bf)`` quota split and the ``q == 1`` shortcut where a group slot
    doubles as a leaf pointer cell rather than gaining a level. Depth counts NODE
    cells below the ROOT, so a flat group is depth 1 and one deref of it costs
    ``depth + 1`` reads (ibid:169-171).
    """
    key = (n, bf)
    hit = _DEPTH_MEMO.get(key)
    if hit is not None:
        return hit

    out: dict = {}
    if n <= bf:
        out[1] = n
    else:
        base, rem = divmod(n, bf)
        quotas = [base + 1] * rem + [base] * (bf - rem)
        for q in quotas:
            if q == 1:
                out[1] = out.get(1, 0) + 1
            else:
                for d, c in subtree_leaf_depths(q, bf).items():
                    out[d + 1] = out.get(d + 1, 0) + c

    _DEPTH_MEMO[key] = out
    return out


def subtree_cells(n: int, bf: int) -> int:
    """SAM cells ``_build_subtree(parent, n)`` allocates and writes.

    Equals ``n`` exactly when the tree is flat, so summing over every pointee
    gives ``distinct_edges + graph_nodes`` at or above the threshold -- one cell
    per edge alias plus one per surviving original head.

    This is a **build-time** footprint. It is not the "Total writes" an o_hnsw log
    reports, which counts the search phase only and moves the other way with bf
    (see the module docstring).
    """
    key = (n, bf)
    hit = _CELLS_MEMO.get(key)
    if hit is not None:
        return hit

    if n <= bf:
        total = n
    else:
        base, rem = divmod(n, bf)
        quotas = [base + 1] * rem + [base] * (bf - rem)
        total = bf
        for q in quotas:
            if q != 1:
                total += subtree_cells(q, bf)

    _CELLS_MEMO[key] = total
    return total


def in_degree_histogram(in_deg, present) -> dict:
    """``{in_degree: how many pointees have it}``, over DiGraph members only."""
    hist: dict = {}
    for d, p in zip(in_deg, present):
        if p:
            hist[d] = hist.get(d, 0) + 1
    return hist


def capacity_report(hist: dict, bf: int) -> dict:
    """What a given fan-out costs: who overflows, and how deep the trees get.

    ``avg_ref_depth`` weights each pointee by its in-degree, because that is how
    many edge-dereferences pass through it -- a hub with in-degree 2,663 sits on
    2,663 derefs, not one. That weighting is the whole reason the max, not a
    percentile, is the right statistic. Below the threshold it is exactly 1.0.
    """
    pointees_over = derefs_over = derefs_total = cells = 0
    depth_weighted = 0.0

    for d, count in hist.items():
        n = d + 1
        derefs_total += d * count
        cells += subtree_cells(n, bf) * count

        if n > bf:
            pointees_over += count
            derefs_over += d * count

        depths = subtree_leaf_depths(n, bf)
        mean_depth = sum(dep * c for dep, c in depths.items()) / n
        depth_weighted += mean_depth * d * count

    return {
        "pointees_over_bf": pointees_over,
        "derefs_over_bf": derefs_over,
        "frac_derefs_over_bf": (derefs_over / derefs_total) if derefs_total else 0.0,
        "avg_ref_depth": (depth_weighted / derefs_total) if derefs_total else 0.0,
        "build_ref_cells": cells,
    }


def block_sizes(stats: dict, branching_factor: int) -> dict:
    """The three candidate models, before clamping.

    ``bs_copies`` is the only binding one. ``bs_nodes`` inverts the *out*-tree
    instead: the fan-out that would let a vertex's neighbour list be reached in
    one hop. In the HNSW driver that is ``--branching-factor`` (o_hnsw.py:2026,
    default 2), which is decoupled from ``--bs``, so it is advisory here and
    ``bf_out_needed`` is the number a user would actually type. It is *not*
    advisory in the ER drivers, where benchmark_single_alg.py:39,50 feeds one
    ``bf`` to both the pointer and the out-tree.

    ``bs_bytes`` is the largest single stored payload: a FanOut header plus
    ``branching_factor + 1`` pointers plus its scalars. It does not depend on the
    dataset at all, which is exactly why the partitioned model does not transfer.
    """
    return {
        "bs_bytes": next_pow2(
            HEADER_BYTES + SLOT_BYTES * (branching_factor + 1) + 4 * BYTES_PER_INT
        ),
        "bs_nodes": bs_for_slots(stats["max_out_degree"]),
        "bf_out_needed": stats["max_out_degree"],
        "bs_copies": bs_for_slots(stats["max_in_degree"] + 1),
        "bs_copies_p99": bs_for_slots(stats["p99_in_degree"] + 1),
    }


def ladder(min_bs: int, max_bs: int):
    """Candidate block sizes: powers of two across the range.

    The floor is o_hnsw's own ``--bs`` default (64, o_hnsw.py:2030) rather than
    probe_partitions' 4096. Sub-threshold points are real data here -- they are
    the deep-tree end of the curve. In the partitioned sweep a sub-threshold block
    means a partition literally does not fit, which is why that probe refuses to
    emit them. 32 is the hard floor: below it ``bf < 2`` and ``_check_bf`` raises.
    """
    bs = next_pow2(max(32, min_bs))
    out = []
    while bs <= max_bs:
        out.append(bs)
        bs *= 2
    return out


def write_csv(out_path: Path, rows):
    """Atomic write, so an interrupted probe never leaves a half CSV behind.

    ``lineterminator="\\n"`` is not cosmetic: csv defaults to CRLF, which leaves a
    trailing ``\\r`` on the last column of every line. The shell drivers look
    columns up by header name, and ``run\\r`` does not match ``run``, so the whole
    sweep reads as "no configs to run".
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


def read_cache_whole(out_path: Path, graph_meta: dict, branching_factor: int):
    """Cached rows for this exact graph and branching factor, or None.

    All-or-nothing: one JSON parse produces the whole ladder, so there is no
    partial-reuse case worth the complexity. ``branching_factor`` is part of the
    guard because it moves ``bs_bytes`` -- probe_partitions' guard omits its own
    ``--model``/``--min-bs``/``--max-bs`` and can silently mix two models in one
    CSV as a result.
    """
    if not out_path.is_file():
        return None

    try:
        with open(out_path, newline="") as f:
            rows = list(csv.DictReader(f))
    except (OSError, csv.Error) as e:
        print(f"  ! ignoring unreadable cache {out_path}: {e}", file=sys.stderr)
        return None

    if not rows:
        return None

    for row in rows:
        same = (
            row.get("graph") == graph_meta["graph"]
            and row.get("graph_size") == str(graph_meta["graph_size"])
            and row.get("graph_mtime") == str(graph_meta["graph_mtime"])
            and row.get("branching_factor") == str(branching_factor)
        )
        if not same:
            return None

    return rows


def print_table(rows, stats, g, sizes, chosen_bs, binding):
    print(f"\n  nodes in graph:  {g['graph_nodes']:,} of {g['ntotal']:,} "
          f"(levels 0-{g['max_level']})")
    print(f"  distinct edges:  {g['distinct_edges']:,}"
          + (f"   ({g['self_loops']:,} self-loops)" if g["self_loops"] else ""))
    print(f"  in-degree:       max {stats['max_in_degree']:,}   "
          f"p99.9 {stats['p999_in_degree']:,}   p99 {stats['p99_in_degree']:,}   "
          f"mean {stats['mean_in_degree']:.1f}")
    print(f"  out-degree:      max {stats['max_out_degree']:,}   "
          f"p99 {stats['p99_out_degree']:,}   mean {stats['mean_out_degree']:.1f}")

    print(f"\n  bs_copies     {fmt_bytes(sizes['bs_copies']):>10}   BINDING -- "
          f"depth-1 reference tree for every pointee")
    print(f"  bs_copies_p99 {fmt_bytes(sizes['bs_copies_p99']):>10}   if the hub "
          f"tail is allowed to go deeper")
    print(f"  bs_nodes      {fmt_bytes(sizes['bs_nodes']):>10}   advisory -- "
          f"out-tree fan-out; use --branching-factor {sizes['bf_out_needed']}, "
          f"not --bs, in the HNSW driver")
    print(f"  bs_bytes      {fmt_bytes(sizes['bs_bytes']):>10}   non-binding -- "
          f"constant; nothing stored grows with n")

    head = (f"{'block size':>11} {'bf':>8} {'over cap':>9} {'% derefs':>9} "
            f"{'avg depth':>10} {'ref cells':>14}")
    print(f"\n{head}")
    print("-" * len(head))
    for row in rows:
        mark = "  <- recommended" if row["recommended"] else ""
        print(f"{fmt_bytes(row['block_size_bytes']):>11} {row['bf_at_bs']:>8,} "
              f"{row['pointees_over_bf']:>9,} "
              f"{100 * row['frac_derefs_over_bf']:>8.2f}% "
              f"{row['avg_ref_depth']:>10.3f} "
              f"{row['build_ref_cells']:>14,}{mark}")

    print(f"\n  recommended --bs {chosen_bs} ({fmt_bytes(chosen_bs)}), "
          f"model={binding}")


def verify_build(graph_path, branching_factor, pointer_type, bs, expected):
    """Check the loader against a real build, exactly, in seconds.

    ``add_vertex`` sets ``ogV_counts[v] = 1`` (oblivious_graph.py:458) and each
    incoming edge increments it (ibid:1239), so after a non-clustered build
    ``ogV_counts[v] == 1 + in_degree(v)`` for every v. That makes the whole degree
    model falsifiable without running a single query -- and it must run before
    one, because search decrements these counters (ibid:1387, 1406).

    ``o_hnsw.py:15`` builds its python-implementation path from
    ``pathlib.Path().resolve().parent`` -- the parent of the *working directory*,
    not of the file -- so the path is inserted here rather than relied upon.
    """
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(HERE.parent / "python-implementation"))
    from o_hnsw import ObliviousHNSW
    from smart_pointer import SmartPointer

    SmartPointer.set_smart_pointer(pointer_type)
    if pointer_type == "multiwriterary":
        SmartPointer.set_branching_factor(bf_for_bs(bs))

    print(f"\n=== --verify-build ({pointer_type}, bs={bs}, "
          f"branching_factor={branching_factor}) ===")
    t0 = time.perf_counter()
    g = ObliviousHNSW(
        branching_factor=branching_factor,
        clustering_method=None,
        use_osam=True,
        make_copies=False,
    )
    # vectors_path=None: _load_vectors only matters for search, and skipping it
    # keeps the check cheap.
    g.load_hnsw_from_json(str(graph_path), None)
    print(f"  built in {time.perf_counter() - t0:.1f}s")

    counts = g.ogV_counts
    checks = (
        ("max in-degree", max(counts.values()) - 1, expected["max_in_degree"]),
        ("sum(ogV_counts)", sum(counts.values()),
         expected["distinct_edges"] + expected["graph_nodes"]),
        ("graph nodes", len(counts), expected["graph_nodes"]),
    )

    ok = True
    for name, got, want in checks:
        if got != want:
            ok = False
        print(f"  {'OK ' if got == want else 'MISMATCH'}  {name:<16} "
              f"probe={want:,}  build={got:,}")

    if not ok:
        raise SystemExit("--verify-build failed: the probe does not model this build")
    print("  ✓ probe matches the build exactly")


def main():
    p = argparse.ArgumentParser(
        description="Predict the ORAM block size for a non-partitioned oblivious "
                    "HNSW run (--partitioning none).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--graph", required=True, help="HNSW graph JSON")
    p.add_argument("--dataset", default=None,
                   help="short name for the CSV and default output path "
                        "(default: the graph's parent directory name)")
    p.add_argument("--out", default=None,
                   help="output CSV (default: sweep/nopart_<dataset>.csv)")
    p.add_argument("--branching-factor", type=int, default=2,
                   help="the --branching-factor the run will use (o_hnsw default 2)")
    p.add_argument("--percentile", choices=PERCENTILES, default="100",
                   help="in-degree statistic feeding block_size_bytes; "
                        "100 = max, matching every partitioned sweep")
    p.add_argument("--min-bs", type=int, default=64, help="block size floor, bytes")
    p.add_argument("--max-bs", type=int, default=1 << 22,
                   help="block size ceiling, bytes")
    p.add_argument("--verify-build", action="store_true",
                   help="build the graph and check ogV_counts against the probe")
    p.add_argument("--verify-max-ntotal", type=int, default=100_000,
                   help="refuse --verify-build above this many nodes")
    p.add_argument("--force", action="store_true", help="ignore the cached CSV")
    p.add_argument("--quiet", action="store_true", help="suppress the table")
    args = p.parse_args()

    graph_path = Path(args.graph)
    if not graph_path.is_file():
        raise SystemExit(f"graph not found: {graph_path}")

    dataset = args.dataset or graph_path.parent.name
    out_path = Path(args.out) if args.out else HERE / "sweep" / f"nopart_{dataset}.csv"

    st = graph_path.stat()
    graph_meta = {
        "dataset": dataset,
        "graph": str(graph_path),
        "graph_size": st.st_size,
        "graph_mtime": int(st.st_mtime),
    }

    if not args.force and not args.verify_build:
        cached = read_cache_whole(out_path, graph_meta, args.branching_factor)
        if cached:
            rec = [r for r in cached if r["recommended"] == "1"]
            print(f"✓ {len(cached)} cached row(s) in {out_path} "
                  f"(--force to recompute)")
            if rec:
                print(f"  recommended --bs {rec[0]['block_size_bytes']}")
            return

    print(f"Loading {graph_path} ...")
    t0 = time.perf_counter()
    g = load_all_levels(graph_path)
    print(f"  {g['graph_nodes']:,} nodes, {g['distinct_edges']:,} distinct edges, "
          f"{time.perf_counter() - t0:.1f}s")

    stats = {}
    stats.update(degree_stats(g["in_deg"], g["present"], "in_degree"))
    stats.update(degree_stats(g["out_deg"], g["present"], "out_degree"))

    sizes = block_sizes(stats, args.branching_factor)

    if args.verify_build:
        if g["ntotal"] > args.verify_max_ntotal:
            raise SystemExit(
                f"--verify-build refused: {g['ntotal']:,} nodes exceeds "
                f"--verify-max-ntotal {args.verify_max_ntotal:,}"
            )
        expected = dict(stats)
        expected["distinct_edges"] = g["distinct_edges"]
        expected["graph_nodes"] = g["graph_nodes"]
        verify_build(graph_path, args.branching_factor, "multiwriterary",
                     sizes["bs_copies"], expected)

    if args.percentile == "100":
        target, binding = sizes["bs_copies"], "copies/max"
    elif args.percentile == "99.9":
        target = bs_for_slots(stats["p999_in_degree"] + 1)
        binding = "copies/p99.9"
    else:
        target, binding = sizes["bs_copies_p99"], "copies/p99"

    floor = next_pow2(max(32, args.min_bs))
    chosen = min(max(target, floor), args.max_bs)
    clamped = int(chosen != target)

    hist = in_degree_histogram(g["in_deg"], g["present"])

    rows = []
    for bs in ladder(args.min_bs, args.max_bs):
        bf = bf_for_bs(bs)
        row = dict(graph_meta)
        row.update(stats)
        row.update(sizes)
        row.update(capacity_report(hist, bf))
        row["ntotal"] = g["ntotal"]
        row["graph_nodes"] = g["graph_nodes"]
        row["distinct_edges"] = g["distinct_edges"]
        row["self_loops"] = g["self_loops"]
        row["max_level"] = g["max_level"]
        row["branching_factor"] = args.branching_factor
        row["mean_in_degree"] = round(stats["mean_in_degree"], 2)
        row["mean_out_degree"] = round(stats["mean_out_degree"], 2)
        row["frac_derefs_over_bf"] = round(row["frac_derefs_over_bf"], 6)
        row["avg_ref_depth"] = round(row["avg_ref_depth"], 4)
        row["block_size_bytes"] = bs
        row["bf_at_bs"] = bf
        row["binding"] = binding
        row["recommended"] = int(bs == chosen)
        row["clamped"] = clamped
        row["run"] = 1
        rows.append(row)

    write_csv(out_path, rows)
    if not args.quiet:
        print_table(rows, stats, g, sizes, chosen, binding)

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20)
    print(f"\n✓ wrote {len(rows)} row(s) to {out_path}   (peak RSS {peak:.2f} GiB)")
    if clamped:
        print(f"  ! target {fmt_bytes(target)} clamped into "
              f"[{fmt_bytes(floor)}, {fmt_bytes(args.max_bs)}]")


if __name__ == "__main__":
    main()
