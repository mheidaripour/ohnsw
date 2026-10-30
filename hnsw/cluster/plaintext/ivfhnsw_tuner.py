#!/usr/bin/env python3
"""
IVF-HNSW tuner: improve recall/performance across datasets.

This tuner is tailored to `ivfhnsw.py`'s implementation:
  - IVF via KMeans (nlist clusters)
  - HNSW over centroids (M, ef_construction fixed in that file's build_index)
  - Query: centroid-HNSW to pick nprobe clusters (ef_search), then exhaustive scan
    of the inverted lists of those clusters.

Budget model:
  The dominant "visit budget" in this implementation is the number of vector
  distance computations in the scanned clusters:
      QueryResult.cluster_distance_computations
  We sweep parameters and report avg/min/max/std of this value per sweep point.

The script builds the index ONCE per dataset per build configuration (nlist, M),
then sweeps query parameters (nprobe, ef_search) without rebuilding.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np

from ivfhnsw import IVFHNSW, load_ivecs  # type: ignore


def load_fvecs_np(path: str) -> np.ndarray:
    """Minimal fvecs loader returning (n,d) float32 numpy array."""
    import struct

    vecs: List[List[float]] = []
    with open(path, "rb") as f:
        while True:
            d_bytes = f.read(4)
            if len(d_bytes) < 4:
                break
            d = struct.unpack("<i", d_bytes)[0]
            x = f.read(d * 4)
            if len(x) < d * 4:
                break
            vec = struct.unpack("<" + "f" * d, x)
            vecs.append(list(vec))
    return np.asarray(vecs, dtype=np.float32)


def recall_at_k(results: List[List[Tuple[int, float]]], gt: List[List[int]], k: int) -> float:
    """Average recall@k (same definition used elsewhere in this repo)."""
    if not results:
        return 0.0
    hits = 0.0
    denom = 0.0
    for res, truth in zip(results, gt):
        if not truth:
            continue
        denom += 1.0
        res_ids = {rid for (rid, _) in res[:k]}
        gt_ids = set(truth[:k])
        hits += (len(res_ids & gt_ids) / float(min(k, len(gt_ids))) if gt_ids else 0.0)
    return hits / denom if denom > 0 else 0.0


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    vectors: str
    queries: str
    gt: str
    nlist: int


DEFAULT_DATASETS: List[DatasetSpec] = [
    DatasetSpec(
        name="1k",
        vectors="data/1k/laion_base.fvecs",
        queries="data/1k/laion_query.fvecs",
        gt="data/1k/gt.ivecs",
        nlist=32,
    ),
    DatasetSpec(
        name="10k",
        vectors="data/10k/laion_base.fvecs",
        queries="data/laion_query.fvecs",
        gt="data/10k/gt.ivecs",
        nlist=100,
    ),
    DatasetSpec(
        name="100k",
        vectors="data/100k/laion_base.fvecs",
        queries="data/laion_query.fvecs",
        gt="data/100k/gt.ivecs",
        nlist=316,
    ),
    DatasetSpec(
        name="1M",
        vectors="data/1m/laion_base.fvecs",
        queries="data/laion_query.fvecs",
        gt="data/1m/gt.ivecs",
        nlist=1000,
    ),
]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_int_list(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def run_sweep_for_dataset(
    ds: DatasetSpec,
    *,
    metric: str,
    num_queries: int,
    k: int,
    nlists: List[int],
    Ms: List[int],
    nprobes: List[int],
    ef_searches: List[int],
    budgets: List[int],
    out_dir: str,
) -> Dict:
    """
    For this dataset, loop over build configs (nlist, M) and query configs (nprobe, ef_search).
    Writes one CSV per dataset containing all sweep rows.
    Writes a best-under-budget JSON per dataset (per build config).
    """

    ensure_dir(out_dir)
    csv_path = os.path.join(out_dir, f"ivfhnsw_tune_{ds.name}.csv")
    json_path = os.path.join(out_dir, f"ivfhnsw_best_{ds.name}.json")

    Q = load_fvecs_np(ds.queries)
    gt = load_ivecs(ds.gt)
    Q = Q[: min(num_queries, len(Q))]
    gt = gt[: len(Q)]

    rows: List[Dict] = []
    best: Dict[str, Dict[str, Dict]] = {}  # build_key -> budget_str -> best_row

    build_points = list(product(nlists, Ms))
    query_points = list(product(nprobes, ef_searches))
    total_points = len(build_points) * len(query_points)
    point_i = 0

    for nlist, M in build_points:
        print(f"\n{'='*90}")
        print(f"Dataset {ds.name}: building index (nlist={nlist}, M={M})")
        print(f"{'='*90}")

        index = IVFHNSW(vectors_path=ds.vectors, metric=metric, nlist=int(nlist))
        t0 = time.time()
        index.build_index(M=int(M))
        build_s = time.time() - t0

        build_key = f"nlist={nlist},M={M}"
        best.setdefault(build_key, {str(b): None for b in budgets})

        for nprobe, ef_search in query_points:
            point_i += 1
            print(
                f"\r  Sweep {point_i}/{total_points}  build[{build_key}]  nprobe={nprobe} ef_search={ef_search}",
                end="",
                flush=True,
            )

            # Clamp nprobe so it never exceeds nlist (avoids wasting work / oddities)
            nprobe_eff = min(int(nprobe), int(nlist))
            ef_eff = max(int(ef_search), nprobe_eff)

            results: List[List[Tuple[int, float]]] = []
            comps_list: List[float] = []
            total_hops_list: List[float] = []
            centroid_hops0_list: List[float] = []
            clusters_searched_list: List[float] = []

            t1 = time.time()
            for qi in range(len(Q)):
                res, qr = index.search(Q[qi], k=k, nprobe=nprobe_eff, ef_search=ef_eff)
                results.append(res)
                comps_list.append(float(qr.cluster_distance_computations))
                total_hops_list.append(float(qr.total_hops))
                clusters_searched_list.append(float(qr.clusters_searched))
                centroid_hops0_list.append(float(qr.centroid_hops_per_layer.get(0, 0)))

            elapsed = time.time() - t1
            qps = (len(Q) / elapsed) if elapsed > 0 else 0.0

            r = recall_at_k(results, gt, k)

            comps = np.asarray(comps_list, dtype=np.float64)
            total_hops = np.asarray(total_hops_list, dtype=np.float64)
            clusters_searched = np.asarray(clusters_searched_list, dtype=np.float64)
            centroid_hops0 = np.asarray(centroid_hops0_list, dtype=np.float64)

            row = {
                "dataset": ds.name,
                "metric": metric,
                "k": int(k),
                "num_queries": int(len(Q)),
                # build config
                "nlist": int(nlist),
                "M": int(M),
                "build_s": float(round(build_s, 3)),
                # query config (effective)
                "nprobe": int(nprobe_eff),
                "ef_search": int(ef_eff),
                # key outcomes
                "recall_at_k": float(r),
                "qps": float(qps),
                # budgets / costs
                "avg_cluster_comps": float(comps.mean()) if comps.size else 0.0,
                "min_cluster_comps": float(comps.min()) if comps.size else 0.0,
                "max_cluster_comps": float(comps.max()) if comps.size else 0.0,
                "std_cluster_comps": float(comps.std()) if comps.size else 0.0,
                "avg_total_hops": float(total_hops.mean()) if total_hops.size else 0.0,
                "avg_clusters_searched": float(clusters_searched.mean()) if clusters_searched.size else 0.0,
                "avg_centroid_hops_L0": float(centroid_hops0.mean()) if centroid_hops0.size else 0.0,
            }
            rows.append(row)

            # Track best under each budget (constraint on avg_cluster_comps)
            for b in budgets:
                if row["avg_cluster_comps"] > float(b):
                    continue
                cur = best[build_key][str(b)]
                if cur is None or row["recall_at_k"] > cur["recall_at_k"]:
                    best[build_key][str(b)] = row

    print()

    # Write CSV
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)

    # Write best JSON
    with open(json_path, "w") as f:
        json.dump(
            {
                "dataset": ds.name,
                "budgets_avg_cluster_comps": budgets,
                "best_by_build_then_budget": best,
                "csv": os.path.basename(csv_path),
            },
            f,
            indent=2,
        )

    print(f"  Wrote {csv_path}")
    print(f"  Wrote {json_path}")

    return {"dataset": ds.name, "csv_path": csv_path, "json_path": json_path}


def main() -> None:
    ap = argparse.ArgumentParser(description="IVF-HNSW tuner (build once, sweep query params)")
    ap.add_argument("--out-dir", type=str, default="log/ivfhnsw_tuning", help="Output directory")

    # datasets
    ap.add_argument("--datasets", type=str, default="all", help="Comma-separated: 1k,10k,100k,1M or 'all'")
    ap.add_argument("--num-queries", type=int, default=100)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--metric", type=str, default="l2", choices=["l2", "ip"])

    # build sweeps
    ap.add_argument("--nlists", type=str, default="", help="Comma-separated nlist values. Empty => per-dataset defaults")
    ap.add_argument("--Ms", type=str, default="16", help="Comma-separated centroid-HNSW M values")

    # query sweeps
    ap.add_argument("--nprobes", type=str, default="1,2,4,8,16,32", help="Comma-separated nprobe values")
    ap.add_argument("--ef-searches", type=str, default="8,16,32,64,128", help="Comma-separated ef_search values (centroid HNSW)")

    # budget thresholds (avg cluster comps)
    ap.add_argument(
        "--budgets",
        type=str,
        default="2000,5000,10000,20000,50000",
        help="Comma-separated thresholds on avg_cluster_comps (distance computations/query)",
    )

    args = ap.parse_args()

    ds_filter = args.datasets.strip()
    if ds_filter.lower() == "all":
        datasets = DEFAULT_DATASETS
    else:
        want = {x.strip() for x in ds_filter.split(",") if x.strip()}
        datasets = [d for d in DEFAULT_DATASETS if d.name in want]
        if not datasets:
            raise SystemExit(f"No datasets matched {want}. Expected one of {[d.name for d in DEFAULT_DATASETS]} or 'all'.")

    Ms = parse_int_list(args.Ms)
    nprobes = parse_int_list(args.nprobes)
    ef_searches = parse_int_list(args.ef_searches)
    budgets = parse_int_list(args.budgets)

    nlists_override = parse_int_list(args.nlists) if args.nlists.strip() else []

    ensure_dir(args.out_dir)
    outputs: Dict[str, Dict[str, str]] = {}

    for ds in datasets:
        nlists = nlists_override if nlists_override else [ds.nlist]
        out = run_sweep_for_dataset(
            ds,
            metric=args.metric,
            num_queries=args.num_queries,
            k=args.k,
            nlists=nlists,
            Ms=Ms,
            nprobes=nprobes,
            ef_searches=ef_searches,
            budgets=budgets,
            out_dir=args.out_dir,
        )
        outputs[ds.name] = {"csv": os.path.basename(out["csv_path"]), "best": os.path.basename(out["json_path"])}

    summary_path = os.path.join(args.out_dir, "ivfhnsw_tune_summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "datasets": [d.name for d in datasets],
                "metric": args.metric,
                "num_queries": args.num_queries,
                "k": args.k,
                "Ms": Ms,
                "nprobes": nprobes,
                "ef_searches": ef_searches,
                "budgets": budgets,
                "outputs": outputs,
            },
            f,
            indent=2,
        )
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()

