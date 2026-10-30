#!/usr/bin/env python3
"""
DiskANN tuner: best recall under hop/visit budget across datasets.

Goal: build index ONCE per dataset, then sweep query-time parameters under a
budget and log results to CSV/JSON for easy comparison.

This script imports `DiskANNIndex` from `diskann.py` and calls
`index.search(...)` directly to avoid rebuilding the index for every sweep
point.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np

from diskann import DiskANNIndex, load_ivecs  # type: ignore


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
    n_clusters: int
    default_nprobe: int


DEFAULT_DATASETS: List[DatasetSpec] = [
    DatasetSpec(
        name="1k",
        vectors="data/1k/laion_base.fvecs",
        queries="data/1k/laion_query.fvecs",
        gt="data/1k/gt.ivecs",
        n_clusters=32,
        default_nprobe=4,
    ),
    DatasetSpec(
        name="10k",
        vectors="data/10k/laion_base.fvecs",
        queries="data/laion_query.fvecs",
        gt="data/10k/gt.ivecs",
        n_clusters=100,
        default_nprobe=8,
    ),
    DatasetSpec(
        name="100k",
        vectors="data/100k/laion_base.fvecs",
        queries="data/laion_query.fvecs",
        gt="data/100k/gt.ivecs",
        n_clusters=316,
        default_nprobe=8,
    ),
    DatasetSpec(
        name="1M",
        vectors="data/1m/laion_base.fvecs",
        queries="data/laion_query.fvecs",
        gt="data/1m/gt.ivecs",
        n_clusters=1000,
        default_nprobe=16,
    ),
]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run_sweep_for_dataset(
    ds: DatasetSpec,
    *,
    metric: str,
    num_queries: int,
    k: int,
    budgets: List[int],
    nprobes: List[int],
    l_centroids: List[int],
    allow_inter_cluster_values: List[bool],
    # build params
    expansion: float,
    R_centroid: int,
    R_cluster: int,
    alpha: float,
    boundary_fraction: float,
    build_L_centroid: int,
    build_L_cluster: int,
    out_dir: str,
) -> Dict:
    print(f"\n{'='*90}")
    print(f"Dataset {ds.name}: building index once")
    print(f"{'='*90}")

    index = DiskANNIndex(
        vectors_path=ds.vectors,
        metric=metric,
        n_clusters=ds.n_clusters,
        expansion_factor=expansion,
        R_centroid=R_centroid,
        R_cluster=R_cluster,
        alpha=alpha,
        boundary_fraction=boundary_fraction,
        build_L_centroid=build_L_centroid,
        build_L_cluster=build_L_cluster,
    )

    t0 = time.time()
    index.build_index()
    build_s = time.time() - t0

    Q = load_fvecs_np(ds.queries)
    gt = load_ivecs(ds.gt)
    Q = Q[: min(num_queries, len(Q))]
    gt = gt[: len(Q)]

    # sweep
    rows: List[Dict] = []
    best_by_budget: Dict[int, Dict] = {}

    total_points = len(budgets) * len(nprobes) * len(l_centroids) * len(allow_inter_cluster_values)
    point_i = 0

    for L_total, nprobe, L_centroid, allow_inter in product(budgets, nprobes, l_centroids, allow_inter_cluster_values):
        point_i += 1
        print(f"\r  Sweep {point_i}/{total_points}  L_total={L_total} nprobe={nprobe} L_centroid={L_centroid} inter={int(allow_inter)}", end="", flush=True)

        results: List[List[Tuple[int, float]]] = []
        sum_total_hops = 0.0
        sum_vectors_visited = 0.0
        sum_clusters_accessed = 0.0
        clusters_accessed_list: List[float] = []

        t1 = time.time()
        for qi in range(len(Q)):
            res, st = index.search(
                Q[qi],
                k=k,
                nprobe=nprobe,
                L_centroid=L_centroid,
                L_cluster=50,  # ignored when L_total is set
                allow_inter_cluster=allow_inter,
                L_total=L_total,
            )
            results.append(res)
            sum_total_hops += float(st["total_hops"])
            sum_vectors_visited += float(st["vectors_visited"])
            ca = float(st["clusters_accessed"])
            sum_clusters_accessed += ca
            clusters_accessed_list.append(ca)

        elapsed = time.time() - t1
        qps = (len(Q) / elapsed) if elapsed > 0 else 0.0

        r = recall_at_k(results, gt, k)
        avg_total_hops = sum_total_hops / float(len(Q))
        avg_vectors_visited = sum_vectors_visited / float(len(Q))
        avg_clusters_accessed = sum_clusters_accessed / float(len(Q))
        min_clusters_accessed = float(min(clusters_accessed_list)) if clusters_accessed_list else 0.0
        max_clusters_accessed = float(max(clusters_accessed_list)) if clusters_accessed_list else 0.0
        std_clusters_accessed = float(np.std(np.asarray(clusters_accessed_list, dtype=np.float64))) if clusters_accessed_list else 0.0

        row = {
            "dataset": ds.name,
            "k": k,
            "num_queries": len(Q),
            "metric": metric,
            "build_s": round(build_s, 3),
            # build params
            "n_clusters": ds.n_clusters,
            "expansion": expansion,
            "R_centroid": R_centroid,
            "R_cluster": R_cluster,
            "alpha": alpha,
            "boundary_fraction": boundary_fraction,
            "build_L_centroid": build_L_centroid,
            "build_L_cluster": build_L_cluster,
            # query params
            "L_total": int(L_total),
            "nprobe": int(nprobe),
            "L_centroid": int(L_centroid),
            "allow_inter_cluster": int(allow_inter),
            # outcomes
            "recall_at_k": float(r),
            "avg_total_hops": float(avg_total_hops),
            "avg_vectors_visited": float(avg_vectors_visited),
            "avg_clusters_accessed": float(avg_clusters_accessed),
            "min_clusters_accessed": float(min_clusters_accessed),
            "max_clusters_accessed": float(max_clusters_accessed),
            "std_clusters_accessed": float(std_clusters_accessed),
            "qps": float(qps),
        }
        rows.append(row)

        # best under each budget, using visit budget as primary constraint.
        # (We also report hops so you can choose hop-budget later.)
        cur_best = best_by_budget.get(L_total)
        if cur_best is None or row["recall_at_k"] > cur_best["recall_at_k"]:
            best_by_budget[L_total] = row

    print()

    # write outputs
    ensure_dir(out_dir)
    csv_path = os.path.join(out_dir, f"diskann_tune_{ds.name}.csv")
    json_path = os.path.join(out_dir, f"diskann_best_{ds.name}.json")

    # CSV
    import csv

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    with open(json_path, "w") as f:
        json.dump(
            {
                "dataset": ds.name,
                "best_by_L_total": {str(k): v for k, v in best_by_budget.items()},
                "csv": os.path.basename(csv_path),
            },
            f,
            indent=2,
        )

    print(f"  Wrote {csv_path}")
    print(f"  Wrote {json_path}")

    return {
        "dataset": ds.name,
        "build_s": build_s,
        "csv_path": csv_path,
        "best_by_budget": best_by_budget,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="DiskANN tuner: maximize recall under visit/hop budget")
    ap.add_argument("--out-dir", type=str, default="log/diskann_tuning", help="Output directory for logs/CSVs")

    # datasets
    ap.add_argument("--datasets", type=str, default="all", help="Comma-separated: 1k,10k,100k,1M or 'all'")
    ap.add_argument("--num-queries", type=int, default=100)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--metric", type=str, default="l2", choices=["l2", "ip"])

    # query sweep knobs
    ap.add_argument("--budgets", type=str, default="200,400,800,1200,2000", help="Comma-separated L_total budgets")
    ap.add_argument("--nprobes", type=str, default="4,8,16", help="Comma-separated nprobe values")
    ap.add_argument("--L-centroids", type=str, default="50,100,200", help="Comma-separated L_centroid values")
    ap.add_argument("--inter", type=str, default="on,off", help="'on', 'off', or 'on,off'")

    # build params (fixed during sweep)
    ap.add_argument("--n-clusters", type=int, default=0, help="Override n_clusters for ALL datasets (0 keeps defaults)")
    ap.add_argument("--expansion", type=float, default=1.3)
    ap.add_argument("--R-centroid", type=int, default=32)
    ap.add_argument("--R-cluster", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=1.2)
    ap.add_argument("--boundary-fraction", type=float, default=0.2)
    ap.add_argument("--build-L-centroid", type=int, default=200)
    ap.add_argument("--build-L-cluster", type=int, default=100)

    args = ap.parse_args()

    ds_filter = args.datasets.strip()
    if ds_filter.lower() == "all":
        datasets = DEFAULT_DATASETS
    else:
        want = {x.strip() for x in ds_filter.split(",") if x.strip()}
        datasets = [d for d in DEFAULT_DATASETS if d.name in want]
        if not datasets:
            raise SystemExit(f"No datasets matched {want}. Expected one of {[d.name for d in DEFAULT_DATASETS]} or 'all'.")

    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]
    nprobes = [int(x) for x in args.nprobes.split(",") if x.strip()]
    l_centroids = [int(x) for x in args.L_centroids.split(",") if x.strip()]
    inter_vals: List[bool] = []
    for tok in [t.strip().lower() for t in args.inter.split(",") if t.strip()]:
        if tok == "on":
            inter_vals.append(True)
        elif tok == "off":
            inter_vals.append(False)
        else:
            raise SystemExit("--inter must be 'on', 'off', or 'on,off'")
    if not inter_vals:
        inter_vals = [True, False]

    out_dir = args.out_dir
    ensure_dir(out_dir)

    all_summaries = []
    for ds in datasets:
        n_clusters = args.n_clusters if args.n_clusters > 0 else ds.n_clusters
        ds_eff = DatasetSpec(
            name=ds.name,
            vectors=ds.vectors,
            queries=ds.queries,
            gt=ds.gt,
            n_clusters=n_clusters,
            default_nprobe=ds.default_nprobe,
        )
        summary = run_sweep_for_dataset(
            ds_eff,
            metric=args.metric,
            num_queries=args.num_queries,
            k=args.k,
            budgets=budgets,
            nprobes=nprobes,
            l_centroids=l_centroids,
            allow_inter_cluster_values=inter_vals,
            expansion=args.expansion,
            R_centroid=args.R_centroid,
            R_cluster=args.R_cluster,
            alpha=args.alpha,
            boundary_fraction=args.boundary_fraction,
            build_L_centroid=args.build_L_centroid,
            build_L_cluster=args.build_L_cluster,
            out_dir=out_dir,
        )
        all_summaries.append(summary)

    # Write a small global summary JSON
    global_path = os.path.join(out_dir, "diskann_tune_summary.json")
    with open(global_path, "w") as f:
        json.dump(
            {
                "datasets": [s["dataset"] for s in all_summaries],
                "budgets": budgets,
                "nprobes": nprobes,
                "L_centroids": l_centroids,
                "inter": [int(x) for x in inter_vals],
                "outputs": {s["dataset"]: {"csv": os.path.basename(s["csv_path"])} for s in all_summaries},
            },
            f,
            indent=2,
        )
    print(f"\nWrote {global_path}")


if __name__ == "__main__":
    main()

