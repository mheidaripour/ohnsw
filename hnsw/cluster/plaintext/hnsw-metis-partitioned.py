#!/usr/bin/env python3
"""
HNSW with METIS balanced graph partitioning (L0 only) and subgraph caching.

Query algorithm matches `hnsw-clustered.py`:
- Upper layers (L>0): standard greedy HNSW descent on the original graph.
- Base layer (L0): standard ef-search with per-partition subgraph caching.

Partitioning uses METIS k-way partitioning on the undirected L0 graph to minimize
edge cuts while balancing partition sizes. Reports the same query/cluster stats
as `hnsw-clustered.py`, a per-layer hop breakdown (avg/min/max/total per HNSW
layer) with an upper-layer vs base-layer cost split, plus METIS-specific
partition quality metrics.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import struct
import time
from collections import defaultdict
from dataclasses import dataclass, field
from heapq import heappush, heappop, heapreplace
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

try:
    import pymetis

    _HAS_METIS = True
except ImportError:
    _HAS_METIS = False
    pymetis = None  # type: ignore


# ----------------------------
# IO helpers
# ----------------------------


def load_fvecs(path: str) -> Tuple[np.ndarray, int, int]:
    """Load vectors from .fvecs format."""
    vectors = []
    with open(path, "rb") as f:
        while True:
            dim_bytes = f.read(4)
            if len(dim_bytes) < 4:
                break
            dim = struct.unpack("<i", dim_bytes)[0]
            vector_bytes = f.read(dim * 4)
            if len(vector_bytes) < dim * 4:
                break
            vector = struct.unpack("<" + "f" * dim, vector_bytes)
            vectors.append(list(vector))

    n = len(vectors)
    d = len(vectors[0]) if n > 0 else 0
    return np.array(vectors, dtype=np.float32), n, d


def load_ivecs(path: str) -> np.ndarray:
    """Load ground truth from .ivecs format."""
    gt = []
    with open(path, "rb") as f:
        while True:
            d_bytes = f.read(4)
            if not d_bytes:
                break
            d = struct.unpack("i", d_bytes)[0]
            neighbors = struct.unpack("i" * d, f.read(4 * d))
            gt.append(list(neighbors))
    return np.array(gt)


def _default_n_partitions(ntotal: int) -> int:
    """Match `hnsw-clustered.py` auto cluster count."""
    return max(2, min(100, int(math.sqrt(ntotal))))


# ----------------------------
# Data structures
# ----------------------------


@dataclass
class PartitionSubgraph:
    """L0 subgraph for one METIS partition (cluster)."""
    partition_id: int
    nodes: Set[int] = field(default_factory=set)
    intra_neighbors: Dict[int, List[int]] = field(default_factory=dict)
    cross_neighbors: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict)


@dataclass
class QueryStats:
    clusters_accessed: int = 0
    cluster_cache_hits: int = 0
    total_hops: int = 0
    hops_per_layer: Dict[int, int] = field(default_factory=dict)
    nodes_visited: int = 0


@dataclass
class AggregateStats:
    total_queries: int = 0
    total_clusters_accessed: int = 0
    total_cache_hits: int = 0
    total_hops: int = 0
    total_nodes_visited: int = 0
    clusters_accessed_per_query: List[int] = field(default_factory=list)
    cache_hits_per_query: List[int] = field(default_factory=list)
    nodes_visited_per_query: List[int] = field(default_factory=list)
    # level -> per-query hop counts (one entry per query that touched the level)
    hops_per_layer: Dict[int, List[int]] = field(default_factory=dict)


@dataclass
class PartitionMetrics:
    """METIS and graph partition quality metrics."""
    n_partitions_requested: int = 0
    n_partitions: int = 0
    edge_cuts: int = 0
    l0_undirected_edges: int = 0
    cut_ratio: float = 0.0
    partition_time_s: float = 0.0
    size_min: int = 0
    size_max: int = 0
    size_mean: float = 0.0
    size_median: float = 0.0
    size_std: float = 0.0
    size_gini: float = 0.0
    size_cv: float = 0.0
    size_max_over_median: float = 0.0
    intra_edges_total: int = 0
    cross_edges_total: int = 0
    intra_fraction: float = 0.0

    def to_dict(self) -> dict:
        return {
            "n_partitions_requested": self.n_partitions_requested,
            "n_partitions": self.n_partitions,
            "edge_cuts": self.edge_cuts,
            "l0_undirected_edges": self.l0_undirected_edges,
            "cut_ratio": self.cut_ratio,
            "partition_time_s": self.partition_time_s,
            "size_min": self.size_min,
            "size_max": self.size_max,
            "size_mean": self.size_mean,
            "size_median": self.size_median,
            "size_std": self.size_std,
            "size_gini": self.size_gini,
            "size_cv": self.size_cv,
            "size_max_over_median": self.size_max_over_median,
            "intra_edges_total": self.intra_edges_total,
            "cross_edges_total": self.cross_edges_total,
            "intra_fraction": self.intra_fraction,
        }


# ----------------------------
# Core implementation
# ----------------------------


class MetisPartitionedHNSW:
    """
    HNSW with METIS L0 partitioning and clustered-style subgraph caching.
    """

    def __init__(self, metric: str = "l2", n_partitions: Optional[int] = None):
        self.metric = metric
        self.n_partitions = n_partitions

        self.max_level: int = 0
        self.entry_point: Optional[int] = None
        self.ntotal: int = 0
        self.vectors: Dict[int, np.ndarray] = {}
        self.level_graphs: Dict[int, Dict[int, List[int]]] = {}

        self.node_to_partition: Dict[int, int] = {}
        self.partition_subgraphs: Dict[int, PartitionSubgraph] = {}
        self.num_partitions: int = 0
        self.partition_metrics: Optional[PartitionMetrics] = None

        self.aggregate_stats = AggregateStats()

        # Populated by run_queries, consumed by print_statistics
        self.last_query_times: List[float] = []
        self.last_total_time: float = 0.0
        self.last_recall: Optional[dict] = None

    # ---- loading ----

    def load_hnsw_json(self, json_path: str) -> dict:
        print(f"Loading HNSW from {json_path}...")
        with open(json_path, "r") as f:
            data = json.load(f)

        meta = data["meta"]
        graph = data["graph"]

        self.ntotal = int(meta["ntotal"])
        self.max_level = int(meta["max_level"])
        self.entry_point = int(meta["entry_point"])

        self.level_graphs = {level: {} for level in range(self.max_level + 1)}
        for node_obj in graph:
            node_id = int(node_obj["node"])
            for level_obj in node_obj["levels"]:
                level = int(level_obj["layer"])
                neighbors = [int(x) for x in level_obj["neighbors"]]
                self.level_graphs[level][node_id] = neighbors

        print(f"  Loaded: {self.ntotal} nodes, {self.max_level + 1} levels")
        print(f"  Entry point: {self.entry_point}")
        return meta

    def load_vectors(self, vectors_path: str) -> None:
        print(f"Loading vectors from {vectors_path}...")
        X, n, d = load_fvecs(vectors_path)
        for i in range(n):
            self.vectors[i] = X[i]
        print(f"  Loaded: {n} vectors of dimension {d}")

    # ---- METIS partitioning on L0 ----

    def build_partitions(self) -> None:
        """Run METIS k-way partitioning on L0 and build cached partition subgraphs."""
        if not _HAS_METIS:
            raise RuntimeError(
                "pymetis is required for METIS partitioning. "
                "Install with: pip install pymetis (requires libmetis)."
            )
        if not self.vectors:
            raise RuntimeError("Vectors must be loaded before building partitions")

        if self.n_partitions is None:
            self.n_partitions = _default_n_partitions(self.ntotal)

        num_parts_requested = int(self.n_partitions)
        print(f"\nBuilding METIS partitions on L0 (n_partitions={num_parts_requested})...")
        start_time = time.time()

        base_layer = self.level_graphs.get(0, {})
        adjacency_list, l0_undirected_edges = self._build_l0_adjacency_for_metis(
            base_layer, self.ntotal
        )
        print(f"  L0 graph: {self.ntotal} nodes, {l0_undirected_edges} undirected edges")

        num_parts = min(num_parts_requested, self.ntotal)
        if self.ntotal < num_parts * 2:
            num_parts = max(2, self.ntotal // 10)
            print(f"  Adjusted to {num_parts} partitions (graph too small)")

        edge_cuts = 0
        try:
            edge_cuts, partition_vector = pymetis.part_graph(num_parts, adjacency=adjacency_list)
            print(f"  METIS edge cuts: {edge_cuts}")
        except Exception as e:
            print(f"  METIS failed ({e}); falling back to round-robin")
            partition_vector = [i % num_parts for i in range(self.ntotal)]
            edge_cuts = self._estimate_edge_cuts(partition_vector, adjacency_list)

        self._assign_partitions(partition_vector)
        intra_total, cross_total = self._build_partition_subgraphs_from_L0(base_layer)

        elapsed = time.time() - start_time
        sizes = np.array(
            [len(sg.nodes) for sg in self.partition_subgraphs.values()],
            dtype=np.int64,
        )
        self._print_partition_size_stats(sizes)
        self.partition_metrics = self._compute_partition_metrics(
            num_parts_requested=num_parts_requested,
            edge_cuts=int(edge_cuts),
            l0_undirected_edges=l0_undirected_edges,
            intra_total=intra_total,
            cross_total=cross_total,
            partition_time_s=elapsed,
            sizes=sizes,
        )
        self._print_partition_metrics()

        print(f"  METIS partitioning completed in {elapsed:.2f}s")

    @staticmethod
    def _build_l0_adjacency_for_metis(
        base_layer: Dict[int, List[int]],
        ntotal: int,
    ) -> Tuple[List[List[int]], int]:
        """Symmetrized undirected adjacency for pymetis (sorted neighbor lists)."""
        adjacency_sets: List[Set[int]] = [set() for _ in range(ntotal)]
        for node_id, neighbors in base_layer.items():
            for neighbor in neighbors:
                if neighbor != node_id:
                    adjacency_sets[node_id].add(neighbor)
                    adjacency_sets[neighbor].add(node_id)

        adjacency_list = [sorted(neighbors) for neighbors in adjacency_sets]
        edge_count = sum(len(adj) for adj in adjacency_list) // 2
        return adjacency_list, edge_count

    def _assign_partitions(self, partition_vector: List[int]) -> None:
        self.node_to_partition.clear()
        self.partition_subgraphs.clear()

        raw: Dict[int, Set[int]] = defaultdict(set)
        for node_id, part_id in enumerate(partition_vector):
            raw[int(part_id)].add(node_id)

        # Reindex to contiguous 0..k-1
        for new_id, nodes in enumerate(raw.values()):
            if not nodes:
                continue
            sg = PartitionSubgraph(partition_id=new_id)
            sg.nodes = set(nodes)
            self.partition_subgraphs[new_id] = sg
            for node_id in sg.nodes:
                self.node_to_partition[node_id] = new_id

        self.num_partitions = len(self.partition_subgraphs)

    def _build_partition_subgraphs_from_L0(
        self, base_layer: Dict[int, List[int]]
    ) -> Tuple[int, int]:
        print("  Building partition subgraphs...")
        for node_id in range(self.ntotal):
            part_id = self.node_to_partition.get(node_id)
            if part_id is None:
                continue
            sg = self.partition_subgraphs[part_id]
            sg.intra_neighbors.setdefault(node_id, [])
            sg.cross_neighbors.setdefault(node_id, [])

        intra_total = 0
        cross_total = 0

        for node_id, neighbors in base_layer.items():
            part_id = self.node_to_partition[node_id]
            sg = self.partition_subgraphs[part_id]

            intra: List[int] = []
            cross: List[Tuple[int, int]] = []
            for nb in neighbors:
                nb_part = self.node_to_partition[nb]
                if nb_part == part_id:
                    intra.append(nb)
                else:
                    cross.append((nb, nb_part))

            sg.intra_neighbors[node_id] = intra
            sg.cross_neighbors[node_id] = cross
            intra_total += len(intra)
            cross_total += len(cross)

        print(f"    Intra-partition edges (directed): {intra_total}")
        print(f"    Cross-partition edges (directed): {cross_total}")
        return intra_total, cross_total

    @staticmethod
    def _estimate_edge_cuts(partition_vector: List[int], adjacency_list: List[List[int]]) -> int:
        cuts = 0
        for u, neighbors in enumerate(adjacency_list):
            pu = partition_vector[u]
            for v in neighbors:
                if u < v and partition_vector[v] != pu:
                    cuts += 1
        return cuts

    def _compute_partition_metrics(
        self,
        num_parts_requested: int,
        edge_cuts: int,
        l0_undirected_edges: int,
        intra_total: int,
        cross_total: int,
        partition_time_s: float,
        sizes: np.ndarray,
    ) -> PartitionMetrics:
        cut_ratio = (edge_cuts / l0_undirected_edges) if l0_undirected_edges > 0 else 0.0
        directed_total = intra_total + cross_total
        intra_fraction = (intra_total / directed_total) if directed_total > 0 else 0.0

        if sizes.size == 0:
            return PartitionMetrics(
                n_partitions_requested=num_parts_requested,
                n_partitions=0,
                edge_cuts=edge_cuts,
                l0_undirected_edges=l0_undirected_edges,
                cut_ratio=cut_ratio,
                partition_time_s=partition_time_s,
                intra_edges_total=intra_total,
                cross_edges_total=cross_total,
                intra_fraction=intra_fraction,
            )

        sizes_sorted = np.sort(sizes)
        mean = float(sizes_sorted.mean())
        median = float(np.percentile(sizes_sorted, 50))
        std = float(np.std(sizes_sorted))
        cv = (std / mean) if mean > 0 else 0.0

        n = float(sizes_sorted.size)
        if mean > 0:
            idx = np.arange(1, sizes_sorted.size + 1, dtype=np.float64)
            gini = float(
                (2.0 * np.sum(idx * sizes_sorted) / (n * np.sum(sizes_sorted))) - ((n + 1.0) / n)
            )
        else:
            gini = 0.0

        max_sz = int(sizes_sorted[-1])
        max_over_median = (max_sz / median) if median > 0 else 0.0

        return PartitionMetrics(
            n_partitions_requested=num_parts_requested,
            n_partitions=int(sizes.size),
            edge_cuts=edge_cuts,
            l0_undirected_edges=l0_undirected_edges,
            cut_ratio=cut_ratio,
            partition_time_s=partition_time_s,
            size_min=int(sizes_sorted[0]),
            size_max=max_sz,
            size_mean=mean,
            size_median=median,
            size_std=std,
            size_gini=gini,
            size_cv=cv,
            size_max_over_median=max_over_median,
            intra_edges_total=intra_total,
            cross_edges_total=cross_total,
            intra_fraction=intra_fraction,
        )

    def _print_partition_size_stats(self, sizes: np.ndarray) -> None:
        if sizes.size == 0:
            print("  Found 0 partitions (unexpected)")
            return

        sizes_sorted = np.sort(sizes)
        mean = float(sizes_sorted.mean())
        median = float(np.percentile(sizes_sorted, 50))
        p90 = float(np.percentile(sizes_sorted, 90))
        p99 = float(np.percentile(sizes_sorted, 99))
        std = float(np.std(sizes_sorted))
        min_sz = int(sizes_sorted[0])
        max_sz = int(sizes_sorted[-1])

        print(f"  Found {self.num_partitions} partitions")
        print(
            f"    Size stats: min={min_sz}, max={max_sz}, mean={mean:.1f}, "
            f"median={median:.1f}, std={std:.1f}"
        )
        print(f"    Percentiles: p90={p90:.1f}, p99={p99:.1f}")
        if median > 0:
            print(f"    Imbalance: max/median={max_sz/median:.2f}")

    def _print_partition_metrics(self) -> None:
        if self.partition_metrics is None:
            return
        m = self.partition_metrics
        print(f"\n  Partition Quality (METIS):")
        print(f"    Edge cuts:              {m.edge_cuts}")
        print(f"    Cut ratio (cuts/|E|):   {m.cut_ratio:.4f}")
        print(f"    Intra edge fraction:    {m.intra_fraction:.4f}")
        print(f"    Balance (gini):         {m.size_gini:.4f}")
        print(f"    Balance (CV):           {m.size_cv:.4f}")

    def _print_cluster_stats(self, detail_stats: Optional[dict] = None) -> None:
        """Cluster statistics with per-cluster min/max/avg and size estimates"""
        if detail_stats is None:
            detail_stats = self.compute_cluster_detail_stats()
        s = detail_stats["summary"]

        def _line(label: str, key: str, show_total: bool = True) -> str:
            d = s[key]
            parts = [
                f"min={d['min']}",
                f"max={d['max']} (cluster {d['max_cluster_id']})",
                f"avg={d['avg']:.1f}",
            ]
            if show_total:
                parts.append(f"total={d['total']}")
            return f"    {label:<24}" + ", ".join(parts)

        print(f"\n  Cluster Statistics:")
        print(f"    Number of clusters:     {self.num_partitions}")
        print(_line("Nodes/cluster:", "n_nodes"))
        print(_line("Intra-cluster edges:", "intra_edges"))
        print(_line("Cross-cluster edges:", "cross_edges"))
        print(_line("Neighbor clusters:", "neighbor_clusters_count", show_total=False))
        print(_line("Graph size (bytes):", "graph_bytes"))
        print(_line("Graph+vectors (bytes):", "graph_plus_vectors_bytes"))
        gb = s["graph_bytes"]
        gvb = s["graph_plus_vectors_bytes"]
        print(f"    Size (MB):              graph avg={gb['avg']/1e6:.2f}, "
              f"graph max={gb['max']/1e6:.2f}, "
              f"graph+vectors avg={gvb['avg']/1e6:.2f}, "
              f"graph+vectors max={gvb['max']/1e6:.2f}")

    def print_graph_structure(self) -> None:
        """Same as `hnsw-clustered.py`."""
        print(f"\n{'='*70}")
        print(f"{'GRAPH STRUCTURE':^70}")
        print(f"{'='*70}")

        print(f"\nBasic Info:")
        print(f"  Total nodes:          {self.ntotal}")
        print(f"  Number of layers:     {self.max_level + 1}")
        print(f"  Entry point:          {self.entry_point}")

        print(f"\nLayer Statistics:")
        print(f"  {'Layer':<8} {'Nodes':<12} {'Edges':<12} {'Avg Degree':<12}")
        print(f"  {'-'*44}")

        total_edges = 0
        for level in range(self.max_level, -1, -1):
            layer = self.level_graphs.get(level, {})
            n_nodes = len(layer)
            n_edges = sum(len(neighbors) for neighbors in layer.values())
            avg_degree = n_edges / n_nodes if n_nodes > 0 else 0
            total_edges += n_edges
            print(f"  L{level:<7} {n_nodes:<12} {n_edges:<12} {avg_degree:<12.2f}")

        print(f"  {'-'*44}")
        print(f"  {'TOTAL':<8} {'':<12} {total_edges:<12}")
        print(f"{'='*70}")

    def compute_cluster_detail_stats(self) -> dict:
        """
        Per-cluster node/edge counts and rough size estimates.

        Size model (4 bytes per integer), identical to `hnsw-clustered.py`:
          graph_bytes = 4 * (n_nodes + intra_edges + 2 * cross_edges)
            - node IDs:               4 bytes each
            - intra adjacency entry:  1 int (neighbor id)
            - cross adjacency entry:  2 ints (neighbor id, neighbor cluster id)
          vectors_bytes = 4 * vector_dim * n_nodes (float32 vectors)
        """
        if not self.partition_subgraphs:
            raise RuntimeError("Partitions must be built before cluster detail stats")

        vector_dim = len(next(iter(self.vectors.values()))) if self.vectors else 0

        per_cluster: Dict[int, dict] = {}
        for cluster_id in sorted(self.partition_subgraphs.keys()):
            sg = self.partition_subgraphs[cluster_id]
            n_nodes = len(sg.nodes)
            intra_edges = sum(len(v) for v in sg.intra_neighbors.values())
            cross_edges = sum(len(v) for v in sg.cross_neighbors.values())

            neighbor_clusters: Set[int] = set()
            for cross_list in sg.cross_neighbors.values():
                for _nbr_id, nbr_cluster_id in cross_list:
                    if nbr_cluster_id != cluster_id:
                        neighbor_clusters.add(int(nbr_cluster_id))

            graph_bytes = 4 * (n_nodes + intra_edges + 2 * cross_edges)
            vectors_bytes = 4 * vector_dim * n_nodes
            per_cluster[int(cluster_id)] = {
                "n_nodes": n_nodes,
                "intra_edges": intra_edges,
                "cross_edges": cross_edges,
                "neighbor_clusters_count": len(neighbor_clusters),
                "graph_bytes": graph_bytes,
                "graph_plus_vectors_bytes": graph_bytes + vectors_bytes,
            }

        def _summary(key: str) -> dict:
            vals = {cid: c[key] for cid, c in per_cluster.items()}
            if not vals:
                return {"min": 0, "max": 0, "avg": 0.0, "total": 0,
                        "min_cluster_id": None, "max_cluster_id": None}
            min_cid = min(vals, key=vals.get)
            max_cid = max(vals, key=vals.get)
            return {
                "min": int(vals[min_cid]),
                "max": int(vals[max_cid]),
                "avg": float(np.mean(list(vals.values()))),
                "total": int(sum(vals.values())),
                "min_cluster_id": int(min_cid),
                "max_cluster_id": int(max_cid),
            }

        summary = {
            key: _summary(key)
            for key in (
                "n_nodes",
                "intra_edges",
                "cross_edges",
                "neighbor_clusters_count",
                "graph_bytes",
                "graph_plus_vectors_bytes",
            )
        }

        return {
            "n_clusters": len(self.partition_subgraphs),
            "vector_dim": int(vector_dim),
            "bytes_per_int": 4,
            "summary": summary,
            "per_cluster": per_cluster,
        }

    def compute_neighbor_cluster_stats(self) -> dict:
        """Same schema as `hnsw-clustered.py`."""
        if not self.partition_subgraphs:
            raise RuntimeError("Partitions must be built before neighbor-cluster stats")

        per_cluster: Dict[int, dict] = {}
        counts: List[int] = []

        for cluster_id, sg in self.partition_subgraphs.items():
            neighbor_clusters: Set[int] = set()
            for cross_list in sg.cross_neighbors.values():
                for _nbr_id, nbr_cluster_id in cross_list:
                    if nbr_cluster_id != cluster_id:
                        neighbor_clusters.add(int(nbr_cluster_id))

            c = len(neighbor_clusters)
            per_cluster[int(cluster_id)] = {
                "neighbor_clusters_count": c,
                "neighbor_clusters": sorted(neighbor_clusters),
            }
            counts.append(c)

        if not counts:
            summary = {"min": 0, "max": 0, "avg": 0.0}
        else:
            summary = {
                "min": int(min(counts)),
                "max": int(max(counts)),
                "avg": float(np.mean(counts)),
            }

        return {
            "n_clusters": int(self.num_partitions),
            "neighbor_clusters_per_cluster": summary,
            "per_cluster": per_cluster,
        }

    # ---- distance + search (mirrors clustered / louvain) ----

    def _distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        if self.metric == "l2":
            diff = vec1 - vec2
            return float(np.dot(diff, diff))
        if self.metric == "ip":
            return float(-np.dot(vec1, vec2))
        raise ValueError(f"Unknown metric: {self.metric}")

    def search(
        self, query: np.ndarray, k: int = 10, ef_search: int = 64
    ) -> Tuple[List[Tuple[int, float]], QueryStats]:
        stats = QueryStats()

        if self.entry_point is None:
            return [], stats

        partition_cache: Dict[int, PartitionSubgraph] = {}
        current_closest = int(self.entry_point)

        for level in range(self.max_level, 0, -1):
            current_closest, level_hops = self._greedy_search_layer(current_closest, query, level)
            stats.hops_per_layer[level] = level_hops
            stats.total_hops += level_hops

        results, base_stats = self._ef_search_with_caching(
            current_closest, query, ef_search, k, partition_cache
        )
        stats.hops_per_layer[0] = base_stats["hops"]
        stats.total_hops += base_stats["hops"]
        stats.clusters_accessed = base_stats["clusters_accessed"]
        stats.cluster_cache_hits = base_stats["cache_hits"]
        stats.nodes_visited = base_stats["nodes_visited"]

        return results[:k], stats

    def _greedy_search_layer(self, entry_node: int, query: np.ndarray, level: int) -> Tuple[int, int]:
        current_node = entry_node
        if current_node not in self.vectors:
            return current_node, 0

        current_dist = self._distance(self.vectors[current_node], query)
        visited = {current_node}
        hops = 0
        improved = True

        while improved:
            improved = False
            neighbors = self.level_graphs.get(level, {}).get(current_node, [])
            for nb in neighbors:
                if nb in visited or nb not in self.vectors:
                    continue
                visited.add(nb)
                hops += 1
                nb_dist = self._distance(self.vectors[nb], query)
                if nb_dist < current_dist:
                    current_node = nb
                    current_dist = nb_dist
                    improved = True
                    break

        return current_node, hops

    def _ef_search_with_caching(
        self,
        entry_node: int,
        query: np.ndarray,
        ef: int,
        k: int,
        partition_cache: Dict[int, PartitionSubgraph],
    ) -> Tuple[List[Tuple[int, float]], dict]:
        visited_nodes: Set[int] = set()
        clusters_accessed: Set[int] = set()
        cache_hits = 0
        hops = 0

        candidates: List[Tuple[float, int]] = []
        dynamic_candidates: List[Tuple[float, int]] = []

        if entry_node not in self.vectors:
            return [], {"hops": 0, "clusters_accessed": 0, "cache_hits": 0, "nodes_visited": 0}

        entry_dist = self._distance(self.vectors[entry_node], query)
        heappush(candidates, (entry_dist, entry_node))
        heappush(dynamic_candidates, (-entry_dist, entry_node))
        visited_nodes.add(entry_node)

        entry_part = self.node_to_partition[entry_node]
        if entry_part not in partition_cache:
            partition_cache[entry_part] = self.partition_subgraphs[entry_part]
            clusters_accessed.add(entry_part)

        while candidates:
            current_dist, current_node = heappop(candidates)

            if len(dynamic_candidates) >= ef:
                worst_dist = -dynamic_candidates[0][0]
                if current_dist > worst_dist:
                    break

            neighbors, was_cache_hit = self._get_cached_neighbors(
                current_node, partition_cache, clusters_accessed
            )
            if was_cache_hit:
                cache_hits += 1

            for nb in neighbors:
                if nb in visited_nodes or nb not in self.vectors:
                    continue
                visited_nodes.add(nb)
                hops += 1
                nb_dist = self._distance(self.vectors[nb], query)

                if len(dynamic_candidates) < ef:
                    heappush(dynamic_candidates, (-nb_dist, nb))
                    heappush(candidates, (nb_dist, nb))
                else:
                    worst_dist = -dynamic_candidates[0][0]
                    if nb_dist < worst_dist:
                        heapreplace(dynamic_candidates, (-nb_dist, nb))
                        heappush(candidates, (nb_dist, nb))

        results = []
        while dynamic_candidates:
            neg_dist, node_id = heappop(dynamic_candidates)
            results.append((node_id, -neg_dist))
        results.sort(key=lambda x: x[1])

        return results, {
            "hops": hops,
            "clusters_accessed": len(clusters_accessed),
            "cache_hits": cache_hits,
            "nodes_visited": len(visited_nodes),
        }

    def _get_cached_neighbors(
        self,
        node_id: int,
        partition_cache: Dict[int, PartitionSubgraph],
        clusters_accessed: Set[int],
    ) -> Tuple[List[int], bool]:
        part_id = self.node_to_partition[node_id]
        was_cache_hit = part_id in partition_cache

        if not was_cache_hit:
            partition_cache[part_id] = self.partition_subgraphs[part_id]
            clusters_accessed.add(part_id)

        sg = partition_cache[part_id]
        neighbors: List[int] = []
        neighbors.extend(sg.intra_neighbors.get(node_id, []))
        for nb_id, _ in sg.cross_neighbors.get(node_id, []):
            neighbors.append(nb_id)
        return neighbors, was_cache_hit

    # ---- benchmarking ----

    def run_queries(
        self,
        queries: np.ndarray,
        k: int = 10,
        ef_search: int = 64,
        ground_truth: Optional[np.ndarray] = None,
        num_queries: Optional[int] = None,
        dataset_name: str = "",
    ) -> dict:
        if num_queries is not None:
            queries = queries[:num_queries]
            if ground_truth is not None:
                ground_truth = ground_truth[:num_queries]

        print(f"\n{'='*80}")
        print(f"Running {len(queries)} queries (k={k}, ef={ef_search})")
        print(f"{'='*80}")

        all_stats: List[QueryStats] = []
        all_results: List[List[Tuple[int, float]]] = []
        query_times: List[float] = []

        start_total = time.time()
        for i, q in enumerate(queries):
            if (i + 1) % 10 == 0 or i == 0:
                print(f"\rQuery {i+1}/{len(queries)}", end="", flush=True)

            start_q = time.time()
            results, stats = self.search(q, k=k, ef_search=ef_search)
            query_times.append(time.time() - start_q)

            all_stats.append(stats)
            all_results.append(results)

        print()
        total_time = time.time() - start_total

        self._compute_aggregate_stats(all_stats)

        recall_metrics = None
        if ground_truth is not None:
            recall_metrics = self._compute_recall(all_results, ground_truth, k)

        self.last_query_times = query_times
        self.last_total_time = total_time
        self.last_recall = recall_metrics
        summary = self.print_statistics(dataset_name=dataset_name)

        n = len(all_stats) if all_stats else 1
        total_cluster_loads = sum(s.clusters_accessed for s in all_stats)
        total_cache_hits = sum(s.cluster_cache_hits for s in all_stats)
        cache_hit_rate = (
            total_cache_hits / (total_cache_hits + total_cluster_loads)
            if (total_cache_hits + total_cluster_loads) > 0
            else 0.0
        )

        return {
            "num_queries": len(queries),
            "total_time": total_time,
            "avg_time_per_query": float(np.mean(query_times)) if query_times else 0.0,
            "qps": (len(queries) / total_time) if total_time > 0 else 0.0,
            "avg_clusters_accessed": float(np.mean([s.clusters_accessed for s in all_stats])) if all_stats else 0.0,
            "avg_cache_hits": float(np.mean([s.cluster_cache_hits for s in all_stats])) if all_stats else 0.0,
            "avg_hops": float(np.mean([s.total_hops for s in all_stats])) if all_stats else 0.0,
            "avg_nodes_visited": float(np.mean([s.nodes_visited for s in all_stats])) if all_stats else 0.0,
            "cache_hit_rate": cache_hit_rate,
            "hop_stats": self.compute_hop_stats(),
            "summary": summary,
            "recall": recall_metrics,
            "n_clusters": self.num_partitions,
            "partition_metrics": (
                self.partition_metrics.to_dict() if self.partition_metrics else None
            ),
        }

    def _compute_aggregate_stats(self, all_stats: List[QueryStats]) -> None:
        hops_per_layer: Dict[int, List[int]] = defaultdict(list)
        for s in all_stats:
            for level, hops in s.hops_per_layer.items():
                hops_per_layer[int(level)].append(int(hops))

        self.aggregate_stats = AggregateStats(
            total_queries=len(all_stats),
            total_clusters_accessed=sum(s.clusters_accessed for s in all_stats),
            total_cache_hits=sum(s.cluster_cache_hits for s in all_stats),
            total_hops=sum(s.total_hops for s in all_stats),
            total_nodes_visited=sum(s.nodes_visited for s in all_stats),
            clusters_accessed_per_query=[s.clusters_accessed for s in all_stats],
            cache_hits_per_query=[s.cluster_cache_hits for s in all_stats],
            nodes_visited_per_query=[s.nodes_visited for s in all_stats],
            hops_per_layer=dict(hops_per_layer),
        )

    def compute_hop_stats(self) -> dict:
        """Per-layer hop statistics across all queries (avg/min/max/total)."""
        stats = self.aggregate_stats
        n = stats.total_queries if stats.total_queries > 0 else 1

        per_layer: Dict[int, dict] = {}
        for level, hops_list in stats.hops_per_layer.items():
            if not hops_list:
                continue
            per_layer[int(level)] = {
                "avg": float(np.mean(hops_list)),
                "min": int(min(hops_list)),
                "max": int(max(hops_list)),
                "total": int(sum(hops_list)),
            }

        upper_total = sum(v["total"] for lvl, v in per_layer.items() if lvl > 0)
        base_total = int(per_layer.get(0, {}).get("total", 0))
        total = upper_total + base_total

        return {
            "num_queries": stats.total_queries,
            "per_layer": per_layer,
            "upper_layer_hops_total": upper_total,
            "upper_layer_hops_avg": upper_total / n,
            "base_layer_hops_total": base_total,
            "base_layer_hops_avg": base_total / n,
            "total_hops": total,
            "avg_hops_per_query": total / n,
        }

    def _compute_recall(
        self, results: List[List[Tuple[int, float]]], ground_truth: np.ndarray, k: int
    ) -> dict:
        recalls_at_1 = []
        recalls_at_10 = []
        recalls_at_k = []

        for res, gt in zip(results, ground_truth):
            result_ids = [r[0] for r in res[:k]]
            gt_ids = set(gt[:k])

            if len(result_ids) >= 1:
                recalls_at_1.append(1.0 if result_ids[0] in gt[:1] else 0.0)

            if len(result_ids) >= 10:
                intersection = len(set(result_ids[:10]) & set(gt[:10]))
                recalls_at_10.append(intersection / 10.0)

            intersection = len(set(result_ids) & gt_ids)
            denom = min(k, len(gt_ids)) if len(gt_ids) > 0 else 1
            recalls_at_k.append(intersection / float(denom))

        return {
            "R@1": float(np.mean(recalls_at_1)) if recalls_at_1 else 0.0,
            "R@10": float(np.mean(recalls_at_10)) if recalls_at_10 else 0.0,
            f"R@{k}": float(np.mean(recalls_at_k)) if recalls_at_k else 0.0,
        }

    def print_statistics(self, dataset_name: str = "") -> dict:
        """
        Hop statistics report, in the same format/naming as the standalone
        METIS hop-counter script.

        Note on the COMMUNITY SEARCH section: this variant performs no centroid
        pre-selection pass (no nprobe), so its community-search hop cost is 0 by
        construction. Communities are loaded on demand during the L0 walk, so
        "communities searched" is the number of distinct communities touched.
        """
        stats = self.aggregate_stats
        num_queries = stats.total_queries
        n = num_queries if num_queries > 0 else 1

        print(f"\n{'='*80}")
        print(f"METIS-HNSW HOP STATISTICS: {dataset_name}" if dataset_name else "METIS-HNSW HOP STATISTICS")
        print(f"{'='*80}")
        print(f"Dataset size: {self.ntotal:,} vectors")
        print(f"Number of communities: {self.num_partitions}")
        print(f"HNSW layers: {self.max_level + 1}")

        # Community statistics
        comm_sizes = [len(sg.nodes) for sg in self.partition_subgraphs.values()]
        if comm_sizes:
            print(
                f"\nCommunity size stats: min={min(comm_sizes)}, max={max(comm_sizes)}, "
                f"avg={sum(comm_sizes)/len(comm_sizes):.1f}"
            )

        # Community search statistics
        print(f"\n{'-'*80}")
        print(f"COMMUNITY SEARCH (finding relevant communities)")
        print(f"{'-'*80}")

        total_comm_hops = 0  # no centroid pre-selection pass in this variant
        avg_comm_hops = total_comm_hops / n
        avg_comms = stats.total_clusters_accessed / n

        print(f"Avg communities searched per query: {avg_comms:.1f}")
        print(f"Avg hops for community search: {avg_comm_hops:.2f}")
        print(f"Total community search hops: {total_comm_hops:,}")
        if stats.clusters_accessed_per_query:
            print(f"Min communities searched per query: {min(stats.clusters_accessed_per_query)}")
            print(f"Max communities searched per query: {max(stats.clusters_accessed_per_query)}")

        # Within-community HNSW statistics
        hop_stats = self.compute_hop_stats()
        per_layer = hop_stats["per_layer"]

        print(f"\n{'-'*80}")
        print(f"WITHIN-COMMUNITY HNSW SEARCH")
        print(f"{'-'*80}")
        print(f"{'Layer':<8} {'Avg Hops':<12} {'Min Hops':<10} {'Max Hops':<10} {'Total Hops':<12}")
        print(f"{'-'*80}")

        for layer in range(self.max_level, -1, -1):
            if layer not in per_layer:
                continue
            v = per_layer[layer]
            print(f"L{layer:<7} {v['avg']:<12.2f} {v['min']:<10} {v['max']:<10} {v['total']:<12,}")

        total_hnsw_hops = hop_stats["total_hops"]
        avg_hnsw_total = hop_stats["avg_hops_per_query"]
        print(f"{'-'*80}")
        print(f"{'Subtotal':<8} {avg_hnsw_total:<12.2f} {'':<10} {'':<10} {total_hnsw_hops:<12,}")

        # Community cache statistics
        print(f"\n{'-'*80}")
        print(f"COMMUNITY CACHE")
        print(f"{'-'*80}")
        loads = stats.total_cache_hits + stats.total_clusters_accessed
        print(f"Total communities loaded: {stats.total_clusters_accessed:,}")
        print(f"Total cache hits: {stats.total_cache_hits:,}")
        print(f"Avg cache hits per query: {stats.total_cache_hits/n:.2f}")
        if loads > 0:
            print(f"Cache hit rate: {stats.total_cache_hits/loads:.4f}")
        print(f"Total nodes visited: {stats.total_nodes_visited:,}")
        print(f"Avg nodes visited per query: {stats.total_nodes_visited/n:.2f}")

        # Timing and recall
        print(f"\n{'-'*80}")
        print(f"TIMING AND RECALL")
        print(f"{'-'*80}")
        total_time = self.last_total_time
        print(f"Total time: {total_time:.3f}s")
        if self.last_query_times:
            print(f"Avg time per query: {np.mean(self.last_query_times)*1000:.3f}ms")
        print(f"QPS: {num_queries/total_time:.1f}" if total_time > 0 else "QPS: N/A")
        if self.last_recall:
            for metric, value in self.last_recall.items():
                print(f"{metric}: {value:.4f}")

        # Total statistics
        print(f"\n{'-'*80}")
        print(f"TOTAL STATISTICS")
        print(f"{'-'*80}")

        total_hops = total_comm_hops + total_hnsw_hops
        avg_total = total_hops / n

        print(f"{'Component':<35} {'Avg/Query':<15} {'Total':<15}")
        print(f"{'-'*80}")
        print(f"{'Community search hops':<35} {avg_comm_hops:<15.2f} {total_comm_hops:<15,}")
        print(f"{'Within-community HNSW hops':<35} {avg_hnsw_total:<15.2f} {total_hnsw_hops:<15,}")
        print(f"{'-'*80}")
        print(f"{'TOTAL':<35} {avg_total:<15.2f} {total_hops:<15,}")
        print(f"{'='*80}\n")

        return {
            'dataset_size': self.ntotal,
            'num_communities': self.num_partitions,
            'hnsw_layers': self.max_level + 1,
            'num_queries': num_queries,
            'avg_comm_hops': avg_comm_hops,
            'avg_hnsw_hops': avg_hnsw_total,
            'avg_total_hops': avg_total,
            'avg_comms_searched': avg_comms,
        }


# ----------------------------
# CLI / batch runners
# ----------------------------


def _timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def _default_log_dir() -> str:
    here = Path(__file__).resolve()
    return str(here.parent / "log" / "hnsw_metis_partitioned")


def run_all_datasets(
    num_queries: int = 100,
    k: int = 10,
    ef_search: int = 64,
    log_dir: Optional[str] = None,
) -> dict:
    """Run on all datasets with n_partitions aligned to `hnsw-clustered.py`."""
    datasets = [
        {
            "name": "1k",
            "json": "data/1k/hnsw_32_40.json",
            "vectors": "data/1k/laion_base.fvecs",
            "queries": "data/1k/laion_query.fvecs",
            "gt": "data/1k/gt.ivecs",
            "n_partitions": 31,
        },
        {
            "name": "10k",
            "json": "data/10k/hnsw_32_40.json",
            "vectors": "data/10k/laion_base.fvecs",
            "queries": "data/laion_query.fvecs",
            "gt": "data/10k/gt.ivecs",
            "n_partitions": 100,
        },
        {
            "name": "100k",
            "json": "data/100k/hnsw_32_40.json",
            "vectors": "data/100k/laion_base.fvecs",
            "queries": "data/laion_query.fvecs",
            "gt": "data/100k/gt.ivecs",
            "n_partitions": 316,
        },
        {
            "name": "1M",
            "json": "data/1m/hnsw_32_40.json",
            "vectors": "data/1m/laion_base.fvecs",
            "queries": "data/laion_query.fvecs",
            "gt": "data/1m/gt.ivecs",
            "n_partitions": 1000,
        },
    ]

    all_results: dict = {}

    for ds in datasets:
        print(f"\n{'#'*70}")
        print(f"{'DATASET: ' + ds['name']:^70}")
        print(f"{'#'*70}")

        try:
            hnsw = MetisPartitionedHNSW(metric="l2", n_partitions=ds["n_partitions"])
            hnsw.load_hnsw_json(ds["json"])
            hnsw.load_vectors(ds["vectors"])
            hnsw.build_partitions()
            cluster_detail_stats = hnsw.compute_cluster_detail_stats()
            hnsw._print_cluster_stats(cluster_detail_stats)

            neighbor_cluster_stats = hnsw.compute_neighbor_cluster_stats()
            print("\n  Neighbor-Cluster Statistics (per cluster):")
            s = neighbor_cluster_stats["neighbor_clusters_per_cluster"]
            print(f"    min={s['min']}, max={s['max']}, avg={s['avg']:.2f}")

            hnsw.print_graph_structure()

            queries, _, _ = load_fvecs(ds["queries"])
            gt = load_ivecs(ds["gt"])

            results = hnsw.run_queries(
                queries=queries,
                k=k,
                ef_search=ef_search,
                ground_truth=gt,
                num_queries=num_queries,
                dataset_name=ds["name"],
            )
            results["neighbor_cluster_stats"] = neighbor_cluster_stats
            results["cluster_detail_stats"] = cluster_detail_stats["summary"]
            all_results[ds["name"]] = results

        except FileNotFoundError as e:
            print(f"  Skipping {ds['name']}: {e}")
        except Exception as e:
            print(f"  Error processing {ds['name']}: {e}")
            import traceback

            traceback.print_exc()

    print(f"\n{'='*100}")
    print(f"{'METIS-HNSW COMPARISON SUMMARY':^100}")
    print(f"{'='*100}")
    print(
        f"{'Dataset':<10} {'Size':>12} {'Communities':>12} {'Comm Hops':>12} "
        f"{'HNSW Hops':>12} {'Total':>12} {'Comms/Q':>10} {'CutRatio':>10} {'R@10':>8}"
    )
    print(f"{'-'*100}")

    for name, res in all_results.items():
        s = res.get("summary") or {}
        r10 = (res.get("recall") or {}).get("R@10", 0.0)
        cut_ratio = (res.get("partition_metrics") or {}).get("cut_ratio", 0.0)
        print(
            f"{name:<10} {s.get('dataset_size', 0):>12,} {s.get('num_communities', 0):>12} "
            f"{s.get('avg_comm_hops', 0.0):>12.2f} {s.get('avg_hnsw_hops', 0.0):>12.2f} "
            f"{s.get('avg_total_hops', 0.0):>12.2f} {s.get('avg_comms_searched', 0.0):>10.1f} "
            f"{cut_ratio:>10.4f} {r10:>8.4f}"
        )

    print(f"{'='*100}\n")

    if log_dir:
        out_path = str(Path(log_dir) / f"hnsw_metis_partitioned_all_{_timestamp()}.json")
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nWrote all-datasets results JSON to {out_path}")

    return all_results


def main() -> None:
    ap = argparse.ArgumentParser(
        description="HNSW with METIS L0 partitioning and clustered-style query/stats"
    )
    ap.add_argument("--json", type=str, default="data/10k/hnsw_graph.json", help="HNSW JSON file")
    ap.add_argument("--vectors", type=str, default="data/10k/laion_base.fvecs", help="Base vectors")
    ap.add_argument("--queries", type=str, default="data/10k/laion_query.fvecs", help="Query vectors")
    ap.add_argument("--gt", type=str, default="data/10k/gt.ivecs", help="Ground truth")
    ap.add_argument(
        "--n-partitions",
        type=int,
        default=None,
        help="Number of METIS partitions (default: sqrt(n), capped at 100)",
    )
    ap.add_argument("--k", type=int, default=10, help="Number of neighbors")
    ap.add_argument("--ef", type=int, default=64, help="ef_search parameter")
    ap.add_argument("--num-queries", type=int, default=100, help="Number of queries to run")
    ap.add_argument("--metric", type=str, default="l2", choices=["l2", "ip"], help="Distance metric")
    ap.add_argument("--run-all", action="store_true", help="Run on all datasets under data/")
    ap.add_argument("--log-dir", type=str, default=_default_log_dir(), help="Directory for JSON outputs")
    ap.add_argument("--run-name", type=str, default=None, help="Optional run name for log filenames")
    ap.add_argument("--name", type=str, default="", help="Dataset name shown in the report header")
    ap.add_argument(
        "--write-neighbor-cluster-json",
        action="store_true",
        help="Write neighbor-cluster stats JSON to log-dir",
    )

    args = ap.parse_args()
    log_dir = _ensure_dir(args.log_dir)
    run_name = args.run_name or ("all" if args.run_all else "single")

    if args.run_all:
        run_all_datasets(
            num_queries=args.num_queries,
            k=args.k,
            ef_search=args.ef,
            log_dir=log_dir,
        )
        return

    hnsw = MetisPartitionedHNSW(metric=args.metric, n_partitions=args.n_partitions)
    hnsw.load_hnsw_json(args.json)
    hnsw.load_vectors(args.vectors)
    hnsw.build_partitions()
    cluster_detail_stats = hnsw.compute_cluster_detail_stats()
    hnsw._print_cluster_stats(cluster_detail_stats)

    neighbor_cluster_stats = hnsw.compute_neighbor_cluster_stats()
    print("\nNeighbor-Cluster Statistics (per cluster):")
    s = neighbor_cluster_stats["neighbor_clusters_per_cluster"]
    print(f"  min={s['min']}, max={s['max']}, avg={s['avg']:.2f}")

    hnsw.print_graph_structure()

    queries, _, _ = load_fvecs(args.queries)
    gt = None
    try:
        gt = load_ivecs(args.gt)
    except FileNotFoundError:
        print("Ground truth not found, skipping recall calculation")

    results = hnsw.run_queries(
        queries=queries,
        k=args.k,
        ef_search=args.ef,
        ground_truth=gt,
        num_queries=args.num_queries,
        dataset_name=args.name,
    )
    results["neighbor_cluster_stats"] = neighbor_cluster_stats
    # summary only; per_cluster is one entry per cluster (1000+ at 1M scale)
    results["cluster_detail_stats"] = cluster_detail_stats["summary"]

    out_path = str(Path(log_dir) / f"hnsw_metis_partitioned_{run_name}_{_timestamp()}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote results JSON to {out_path}")

    if args.write_neighbor_cluster_json:
        nc_path = str(
            Path(log_dir) / f"hnsw_metis_neighbor_clusters_{run_name}_{_timestamp()}.json"
        )
        with open(nc_path, "w") as f:
            json.dump(neighbor_cluster_stats, f, indent=2)
        print(f"Wrote neighbor-cluster stats JSON to {nc_path}")


if __name__ == "__main__":
    main()
