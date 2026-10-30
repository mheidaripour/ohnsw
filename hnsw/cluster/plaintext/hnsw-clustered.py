#!/usr/bin/env python3
"""
HNSW with K-means Clustering and Subgraph Caching

This implements:
1. K-means clustering on the dataset
2. Packing HNSW base layer nodes into cluster subgraphs
3. Caching entire subgraphs when accessing any node in a cluster
4. Tracking cluster access statistics during query execution
"""

import json
import time
import math
import struct
import argparse
import os
import sys
import datetime as _dt
import contextlib
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from heapq import heappush, heappop, heapreplace
from dataclasses import dataclass, field
from collections import defaultdict
from itertools import chain

import numpy as np
from sklearn.cluster import KMeans


@dataclass
class ClusterSubgraph:
    """Represents a cluster subgraph from the base layer"""
    cluster_id: int
    nodes: Set[int] = field(default_factory=set)
    # node_id -> list of neighbor_ids (intra-cluster only)
    intra_cluster_neighbors: Dict[int, List[int]] = field(default_factory=dict)
    # node_id -> list of (neighbor_id, neighbor_cluster_id) for cross-cluster edges
    cross_cluster_neighbors: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict)
    vectors: Dict[int, np.ndarray] = field(default_factory=dict)


@dataclass
class QueryStats:
    """Statistics for a single query"""
    clusters_accessed: int = 0
    cluster_cache_hits: int = 0
    total_hops: int = 0
    hops_per_layer: Dict[int, int] = field(default_factory=dict)
    nodes_visited: int = 0


@dataclass
class AggregateStats:
    """Aggregate statistics across all queries"""
    total_queries: int = 0
    total_clusters_accessed: int = 0
    total_cache_hits: int = 0
    total_hops: int = 0
    total_nodes_visited: int = 0
    clusters_accessed_per_query: List[int] = field(default_factory=list)
    cache_hits_per_query: List[int] = field(default_factory=list)
    nodes_visited_per_query: List[int] = field(default_factory=list)


@dataclass
class ClusterMetrics:
    """K-means clustering and graph partition quality metrics.

    Mirrors `PartitionMetrics` in `hnsw-metis-partitioned.py` so the two
    experiments produce directly comparable reports.
    """
    n_clusters_requested: int = 0
    n_clusters: int = 0
    edge_cuts: int = 0
    l0_undirected_edges: int = 0
    cut_ratio: float = 0.0
    clustering_time_s: float = 0.0
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
            "n_clusters_requested": self.n_clusters_requested,
            "n_clusters": self.n_clusters,
            "edge_cuts": self.edge_cuts,
            "l0_undirected_edges": self.l0_undirected_edges,
            "cut_ratio": self.cut_ratio,
            "clustering_time_s": self.clustering_time_s,
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


def load_fvecs(path: str) -> Tuple[np.ndarray, int, int]:
    """Load vectors from .fvecs format"""
    vectors = []
    with open(path, 'rb') as f:
        while True:
            dim_bytes = f.read(4)
            if len(dim_bytes) < 4:
                break
            dim = struct.unpack('<i', dim_bytes)[0]
            vector_bytes = f.read(dim * 4)
            if len(vector_bytes) < dim * 4:
                break
            vector = struct.unpack('<' + 'f' * dim, vector_bytes)
            vectors.append(list(vector))
    
    n = len(vectors)
    d = len(vectors[0]) if n > 0 else 0
    return np.array(vectors, dtype=np.float32), n, d


def load_ivecs(path: str) -> np.ndarray:
    """Load ground truth from .ivecs format"""
    gt = []
    with open(path, 'rb') as f:
        while True:
            d_bytes = f.read(4)
            if not d_bytes:
                break
            d = struct.unpack('i', d_bytes)[0]
            neighbors = struct.unpack('i' * d, f.read(4 * d))
            gt.append(list(neighbors))
    return np.array(gt)


class ClusteredHNSW:
    """
    HNSW with K-means clustering and subgraph caching.
    
    The base layer is partitioned into clusters, and entire cluster subgraphs
    are cached when any node in the cluster is accessed during search.
    """
    
    def __init__(self, n_clusters: int = None, metric: str = 'l2'):
        self.n_clusters = n_clusters
        self.metric = metric
        
        # HNSW structure
        self.max_level = 0
        self.entry_point = None
        self.ntotal = 0
        self.vectors: Dict[int, np.ndarray] = {}
        
        # Level graphs: level -> {node_id -> [neighbor_ids]}
        self.level_graphs: Dict[int, Dict[int, List[int]]] = {}
        
        # Clustering structures
        self.cluster_labels: Dict[int, int] = {}  # node_id -> cluster_id
        self.cluster_subgraphs: Dict[int, ClusterSubgraph] = {}  # cluster_id -> subgraph
        self.cluster_centers: np.ndarray = None
        self.cluster_metrics: Optional[ClusterMetrics] = None
        
        # Statistics
        self.aggregate_stats = AggregateStats()
    
    def load_hnsw_json(self, json_path: str) -> dict:
        """Load HNSW graph from JSON file"""
        print(f"Loading HNSW from {json_path}...")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        meta = data['meta']
        graph = data['graph']
        
        self.ntotal = int(meta['ntotal'])
        self.max_level = int(meta['max_level'])
        self.entry_point = int(meta['entry_point'])
        
        # Build level graphs
        self.level_graphs = {level: {} for level in range(self.max_level + 1)}
        
        for node_obj in graph:
            node_id = int(node_obj['node'])
            for level_obj in node_obj['levels']:
                level = int(level_obj['layer'])
                neighbors = [int(x) for x in level_obj['neighbors']]
                self.level_graphs[level][node_id] = neighbors
        
        print(f"  Loaded: {self.ntotal} nodes, {self.max_level + 1} levels")
        print(f"  Entry point: {self.entry_point}")
        
        return meta
    
    def load_vectors(self, vectors_path: str):
        """Load base vectors"""
        print(f"Loading vectors from {vectors_path}...")
        X, n, d = load_fvecs(vectors_path)
        
        for i in range(n):
            self.vectors[i] = X[i]
        
        print(f"  Loaded: {n} vectors of dimension {d}")
    
    def build_clusters(self):
        """Build K-means clusters and pack base layer into cluster subgraphs"""
        if not self.vectors:
            raise RuntimeError("Vectors must be loaded before clustering")
        
        # Determine number of clusters
        if self.n_clusters is None:
            self.n_clusters = max(2, min(self.ntotal, int(math.sqrt(self.ntotal))))
        
        n_clusters_requested = int(self.n_clusters)
        print(f"\nBuilding K-means clusters (n_clusters={n_clusters_requested})...")
        start_time = time.time()
        
        # Prepare vectors for clustering
        node_ids = list(range(self.ntotal))
        X = np.array([self.vectors[i] for i in node_ids], dtype=np.float32)
        
        # Run K-means
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        self.cluster_centers = kmeans.cluster_centers_
        
        # Store cluster assignments
        for i, node_id in enumerate(node_ids):
            self.cluster_labels[node_id] = int(labels[i])
        
        base_layer = self.level_graphs.get(0, {})
        l0_undirected_edges, edge_cuts = self._compute_l0_cut_stats(base_layer)
        print(f"  L0 graph: {self.ntotal} nodes, {l0_undirected_edges} undirected edges")
        print(f"  Edge cuts: {edge_cuts}")
        
        # Build cluster subgraphs from base layer
        intra_total, cross_total = self._build_cluster_subgraphs()
        
        build_time = time.time() - start_time
        sizes = np.array(
            [len(sg.nodes) for sg in self.cluster_subgraphs.values()],
            dtype=np.int64,
        )
        self._print_cluster_size_stats(sizes)
        self.cluster_metrics = self._compute_cluster_metrics(
            n_clusters_requested=n_clusters_requested,
            edge_cuts=edge_cuts,
            l0_undirected_edges=l0_undirected_edges,
            intra_total=intra_total,
            cross_total=cross_total,
            clustering_time_s=build_time,
            sizes=sizes,
        )
        self._print_cluster_quality()
        
        print(f"  K-means clustering completed in {build_time:.2f}s")
    
    def _compute_l0_cut_stats(self, base_layer: Dict[int, List[int]]) -> Tuple[int, int]:
        """
        Count undirected L0 edges and how many of them cross cluster boundaries.

        Uses the same symmetrized-graph definition as the METIS script (each
        unordered pair counted once, self-loops dropped), so `cut_ratio` is
        comparable between the two experiments.
        """
        if not base_layer:
            return 0, 0

        degrees = np.fromiter(
            (len(v) for v in base_layer.values()), dtype=np.int64, count=len(base_layer)
        )
        n_directed = int(degrees.sum())
        if n_directed == 0:
            return 0, 0

        sources = np.repeat(
            np.fromiter(base_layer.keys(), dtype=np.int64, count=len(base_layer)), degrees
        )
        targets = np.fromiter(
            chain.from_iterable(base_layer.values()), dtype=np.int64, count=n_directed
        )

        keep = sources != targets
        sources = sources[keep]
        targets = targets[keep]

        lo = np.minimum(sources, targets)
        hi = np.maximum(sources, targets)
        del sources, targets

        # Encode each unordered pair as a single integer so np.unique dedupes
        # reciprocal edges without materializing Python-level sets.
        stride = int(max(self.ntotal, int(hi.max()) + 1))
        pairs = np.unique(lo * stride + hi)
        del lo, hi

        labels = np.full(stride, -1, dtype=np.int64)
        for node_id, cluster_id in self.cluster_labels.items():
            labels[node_id] = cluster_id

        cuts = int(np.count_nonzero(labels[pairs // stride] != labels[pairs % stride]))
        return int(pairs.size), cuts
    
    def _build_cluster_subgraphs(self) -> Tuple[int, int]:
        """Build cluster subgraphs from base layer edges"""
        print("  Building cluster subgraphs...")
        
        # Initialize cluster subgraphs
        for cluster_id in range(self.n_clusters):
            self.cluster_subgraphs[cluster_id] = ClusterSubgraph(cluster_id=cluster_id)
        
        # Assign nodes to clusters
        for node_id, cluster_id in self.cluster_labels.items():
            self.cluster_subgraphs[cluster_id].nodes.add(node_id)
            self.cluster_subgraphs[cluster_id].vectors[node_id] = self.vectors[node_id]
        
        # Build intra-cluster and cross-cluster edges from base layer (level 0)
        base_layer = self.level_graphs.get(0, {})
        
        intra_total = 0
        cross_total = 0
        
        for node_id, neighbors in base_layer.items():
            node_cluster = self.cluster_labels[node_id]
            subgraph = self.cluster_subgraphs[node_cluster]
            
            intra_neighbors = []
            cross_neighbors = []
            
            for neighbor_id in neighbors:
                neighbor_cluster = self.cluster_labels[neighbor_id]
                
                if neighbor_cluster == node_cluster:
                    # Intra-cluster edge
                    intra_neighbors.append(neighbor_id)
                else:
                    # Cross-cluster edge
                    cross_neighbors.append((neighbor_id, neighbor_cluster))
            
            subgraph.intra_cluster_neighbors[node_id] = intra_neighbors
            subgraph.cross_cluster_neighbors[node_id] = cross_neighbors
            intra_total += len(intra_neighbors)
            cross_total += len(cross_neighbors)
        
        print(f"    Intra-cluster edges (directed): {intra_total}")
        print(f"    Cross-cluster edges (directed): {cross_total}")
        return intra_total, cross_total
    
    def _compute_cluster_metrics(self, n_clusters_requested: int, edge_cuts: int,
                                 l0_undirected_edges: int, intra_total: int,
                                 cross_total: int, clustering_time_s: float,
                                 sizes: np.ndarray) -> ClusterMetrics:
        cut_ratio = (edge_cuts / l0_undirected_edges) if l0_undirected_edges > 0 else 0.0
        directed_total = intra_total + cross_total
        intra_fraction = (intra_total / directed_total) if directed_total > 0 else 0.0
        
        if sizes.size == 0:
            return ClusterMetrics(
                n_clusters_requested=n_clusters_requested,
                n_clusters=0,
                edge_cuts=edge_cuts,
                l0_undirected_edges=l0_undirected_edges,
                cut_ratio=cut_ratio,
                clustering_time_s=clustering_time_s,
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
        
        return ClusterMetrics(
            n_clusters_requested=n_clusters_requested,
            n_clusters=int(sizes.size),
            edge_cuts=edge_cuts,
            l0_undirected_edges=l0_undirected_edges,
            cut_ratio=cut_ratio,
            clustering_time_s=clustering_time_s,
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
    
    def _print_cluster_size_stats(self, sizes: np.ndarray):
        if sizes.size == 0:
            print("  Found 0 clusters (unexpected)")
            return
        
        sizes_sorted = np.sort(sizes)
        mean = float(sizes_sorted.mean())
        median = float(np.percentile(sizes_sorted, 50))
        p90 = float(np.percentile(sizes_sorted, 90))
        p99 = float(np.percentile(sizes_sorted, 99))
        std = float(np.std(sizes_sorted))
        min_sz = int(sizes_sorted[0])
        max_sz = int(sizes_sorted[-1])
        
        print(f"  Found {sizes.size} clusters")
        print(f"    Size stats: min={min_sz}, max={max_sz}, mean={mean:.1f}, "
              f"median={median:.1f}, std={std:.1f}")
        print(f"    Percentiles: p90={p90:.1f}, p99={p99:.1f}")
        if median > 0:
            print(f"    Imbalance: max/median={max_sz/median:.2f}")
    
    def _print_cluster_quality(self):
        if self.cluster_metrics is None:
            return
        m = self.cluster_metrics
        print(f"\n  Cluster Quality (K-means):")
        print(f"    Edge cuts:              {m.edge_cuts}")
        print(f"    Cut ratio (cuts/|E|):   {m.cut_ratio:.4f}")
        print(f"    Intra edge fraction:    {m.intra_fraction:.4f}")
        print(f"    Balance (gini):         {m.size_gini:.4f}")
        print(f"    Balance (CV):           {m.size_cv:.4f}")
    
    def compute_cluster_detail_stats(self) -> dict:
        """
        Per-cluster node/edge counts and rough size estimates.

        Size model (4 bytes per integer), identical to the METIS script:
          graph_bytes = 4 * (n_nodes + intra_edges + 2 * cross_edges)
            - node IDs:               4 bytes each
            - intra adjacency entry:  1 int (neighbor id)
            - cross adjacency entry:  2 ints (neighbor id, neighbor cluster id)
          vectors_bytes = 4 * vector_dim * n_nodes (float32 vectors)
        """
        if not self.cluster_subgraphs:
            raise RuntimeError("Clusters must be built before cluster detail stats")
        
        vector_dim = len(next(iter(self.vectors.values()))) if self.vectors else 0
        
        per_cluster: Dict[int, dict] = {}
        for cluster_id in sorted(self.cluster_subgraphs.keys()):
            sg = self.cluster_subgraphs[cluster_id]
            n_nodes = len(sg.nodes)
            intra_edges = sum(len(v) for v in sg.intra_cluster_neighbors.values())
            cross_edges = sum(len(v) for v in sg.cross_cluster_neighbors.values())
            
            neighbor_clusters: Set[int] = set()
            for cross_list in sg.cross_cluster_neighbors.values():
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
            "n_clusters": len(self.cluster_subgraphs),
            "vector_dim": int(vector_dim),
            "bytes_per_int": 4,
            "summary": summary,
            "per_cluster": per_cluster,
        }
    
    def _print_cluster_stats(self, detail_stats: Optional[dict] = None):
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
        print(f"    Number of clusters:     {len(self.cluster_subgraphs)}")
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
    
    def _distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate distance between vectors"""
        if self.metric == 'l2':
            diff = vec1 - vec2
            return float(np.dot(diff, diff))
        elif self.metric == 'ip':
            return float(-np.dot(vec1, vec2))
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def search(self, query: np.ndarray, k: int = 10, ef_search: int = 64) -> Tuple[List[Tuple[int, float]], QueryStats]:
        """
        Search for k nearest neighbors with cluster caching.
        
        Returns:
            - List of (node_id, distance) tuples
            - QueryStats object with access statistics
        """
        stats = QueryStats()
        
        if self.entry_point is None:
            return [], stats
        
        # Initialize cluster cache for this query
        cluster_cache: Dict[int, ClusterSubgraph] = {}
        
        current_closest = self.entry_point
        
        # Phase 1: Greedy search through upper layers (no clustering)
        for level in range(self.max_level, 0, -1):
            current_closest, level_hops = self._greedy_search_layer(
                current_closest, query, level
            )
            stats.hops_per_layer[level] = level_hops
            stats.total_hops += level_hops
        
        # Phase 2: ef-search at base layer with cluster caching
        results, base_stats = self._ef_search_with_caching(
            current_closest, query, ef_search, k, cluster_cache
        )
        
        stats.hops_per_layer[0] = base_stats['hops']
        stats.total_hops += base_stats['hops']
        stats.clusters_accessed = base_stats['clusters_accessed']
        stats.cluster_cache_hits = base_stats['cache_hits']
        stats.nodes_visited = base_stats['nodes_visited']
        
        return results[:k], stats
    
    def _greedy_search_layer(self, entry_node: int, query: np.ndarray, level: int) -> Tuple[int, int]:
        """Greedy search at upper layer (no clustering)"""
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
            
            for neighbor_id in neighbors:
                if neighbor_id in visited or neighbor_id not in self.vectors:
                    continue
                
                visited.add(neighbor_id)
                hops += 1
                neighbor_dist = self._distance(self.vectors[neighbor_id], query)
                
                if neighbor_dist < current_dist:
                    current_node = neighbor_id
                    current_dist = neighbor_dist
                    improved = True
                    break
        
        return current_node, hops
    
    def _ef_search_with_caching(self, entry_node: int, query: np.ndarray, 
                                 ef: int, k: int,
                                 cluster_cache: Dict[int, ClusterSubgraph]) -> Tuple[List[Tuple[int, float]], dict]:
        """
        ef-search at base layer with cluster subgraph caching.
        
        When accessing a node, if its cluster is not cached, load the entire
        cluster subgraph into cache. Subsequent accesses to nodes in the same
        cluster are served from cache.
        """
        visited_nodes = set()
        clusters_accessed = set()
        cache_hits = 0
        hops = 0
        
        candidates = []  # min-heap for exploration
        dynamic_candidates = []  # max-heap for top-ef results
        
        # Initialize with entry node
        if entry_node not in self.vectors:
            return [], {'hops': 0, 'clusters_accessed': 0, 'cache_hits': 0, 'nodes_visited': 0}
        
        entry_dist = self._distance(self.vectors[entry_node], query)
        heappush(candidates, (entry_dist, entry_node))
        heappush(dynamic_candidates, (-entry_dist, entry_node))
        visited_nodes.add(entry_node)
        
        # Cache the entry node's cluster
        entry_cluster = self.cluster_labels[entry_node]
        if entry_cluster not in cluster_cache:
            cluster_cache[entry_cluster] = self.cluster_subgraphs[entry_cluster]
            clusters_accessed.add(entry_cluster)
        
        while candidates:
            current_dist, current_node = heappop(candidates)
            
            # Early termination
            if len(dynamic_candidates) >= ef:
                worst_dist = -dynamic_candidates[0][0]
                if current_dist > worst_dist:
                    break
            
            # Get neighbors with cluster caching
            neighbors, was_cache_hit = self._get_cached_neighbors(
                current_node, cluster_cache, clusters_accessed
            )
            
            if was_cache_hit:
                cache_hits += 1
            
            for neighbor_id in neighbors:
                if neighbor_id in visited_nodes or neighbor_id not in self.vectors:
                    continue
                
                visited_nodes.add(neighbor_id)
                hops += 1
                neighbor_dist = self._distance(self.vectors[neighbor_id], query)
                
                # Update candidates
                if len(dynamic_candidates) < ef:
                    heappush(dynamic_candidates, (-neighbor_dist, neighbor_id))
                    heappush(candidates, (neighbor_dist, neighbor_id))
                else:
                    worst_dist = -dynamic_candidates[0][0]
                    if neighbor_dist < worst_dist:
                        heapreplace(dynamic_candidates, (-neighbor_dist, neighbor_id))
                        heappush(candidates, (neighbor_dist, neighbor_id))
        
        # Extract results
        results = []
        while dynamic_candidates:
            neg_dist, node_id = heappop(dynamic_candidates)
            results.append((node_id, -neg_dist))
        
        results.sort(key=lambda x: x[1])
        
        return results, {
            'hops': hops,
            'clusters_accessed': len(clusters_accessed),
            'cache_hits': cache_hits,
            'nodes_visited': len(visited_nodes)
        }
    
    def _get_cached_neighbors(self, node_id: int, 
                               cluster_cache: Dict[int, ClusterSubgraph],
                               clusters_accessed: Set[int]) -> Tuple[List[int], bool]:
        """
        Get neighbors for a node, loading cluster subgraph into cache if needed.
        
        Returns:
            - List of neighbor IDs (both intra and cross-cluster)
            - Whether this was a cache hit
        """
        cluster_id = self.cluster_labels[node_id]
        
        # Check if cluster is already cached
        was_cache_hit = cluster_id in cluster_cache
        
        if not was_cache_hit:
            # Load entire cluster subgraph into cache
            cluster_cache[cluster_id] = self.cluster_subgraphs[cluster_id]
            clusters_accessed.add(cluster_id)
        
        # Get neighbors from cached subgraph
        subgraph = cluster_cache[cluster_id]
        neighbors = []
        
        # Add intra-cluster neighbors
        neighbors.extend(subgraph.intra_cluster_neighbors.get(node_id, []))
        
        # Add cross-cluster neighbors (just the neighbor IDs)
        for neighbor_id, _ in subgraph.cross_cluster_neighbors.get(node_id, []):
            neighbors.append(neighbor_id)
        
        return neighbors, was_cache_hit
    
    def run_queries(self, queries: np.ndarray, k: int = 10, ef_search: int = 64,
                    ground_truth: np.ndarray = None, num_queries: int = None) -> dict:
        """Run queries and collect statistics"""
        if num_queries is not None:
            queries = queries[:num_queries]
            if ground_truth is not None:
                ground_truth = ground_truth[:num_queries]
        
        print(f"\n{'='*70}")
        print(f"Running {len(queries)} queries (k={k}, ef={ef_search})")
        print(f"{'='*70}")
        
        all_stats = []
        all_results = []
        query_times = []
        
        start_total = time.time()
        
        for i, query in enumerate(queries):
            if (i + 1) % 10 == 0 or i == 0:
                print(f"\rQuery {i+1}/{len(queries)}", end="", flush=True)
            
            start_q = time.time()
            results, stats = self.search(query, k=k, ef_search=ef_search)
            query_times.append(time.time() - start_q)
            
            all_stats.append(stats)
            all_results.append(results)
        
        print()
        total_time = time.time() - start_total
        
        # Compute aggregate statistics
        self._compute_aggregate_stats(all_stats)
        
        # Compute recall if ground truth provided
        recall_metrics = None
        if ground_truth is not None:
            recall_metrics = self._compute_recall(all_results, ground_truth, k)
        
        # Print results
        self._print_query_results(query_times, total_time, recall_metrics)
        
        total_cluster_loads = sum(s.clusters_accessed for s in all_stats)
        total_cache_hits = sum(s.cluster_cache_hits for s in all_stats)
        cache_hit_rate = (
            total_cache_hits / (total_cache_hits + total_cluster_loads)
            if (total_cache_hits + total_cluster_loads) > 0
            else 0.0
        )
        
        return {
            'num_queries': len(queries),
            'total_time': total_time,
            'avg_time_per_query': np.mean(query_times),
            'qps': len(queries) / total_time,
            'avg_clusters_accessed': np.mean([s.clusters_accessed for s in all_stats]),
            'avg_cache_hits': np.mean([s.cluster_cache_hits for s in all_stats]),
            'avg_hops': np.mean([s.total_hops for s in all_stats]),
            'avg_nodes_visited': float(np.mean([s.nodes_visited for s in all_stats])) if all_stats else 0.0,
            'cache_hit_rate': cache_hit_rate,
            'recall': recall_metrics,
            'n_clusters': len(self.cluster_subgraphs),
            'cluster_metrics': (
                self.cluster_metrics.to_dict() if self.cluster_metrics else None
            ),
        }
    
    def _compute_aggregate_stats(self, all_stats: List[QueryStats]):
        """Compute aggregate statistics"""
        self.aggregate_stats = AggregateStats(
            total_queries=len(all_stats),
            total_clusters_accessed=sum(s.clusters_accessed for s in all_stats),
            total_cache_hits=sum(s.cluster_cache_hits for s in all_stats),
            total_hops=sum(s.total_hops for s in all_stats),
            total_nodes_visited=sum(s.nodes_visited for s in all_stats),
            clusters_accessed_per_query=[s.clusters_accessed for s in all_stats],
            cache_hits_per_query=[s.cluster_cache_hits for s in all_stats],
            nodes_visited_per_query=[s.nodes_visited for s in all_stats]
        )
    
    def _compute_recall(self, results: List[List[Tuple[int, float]]], 
                        ground_truth: np.ndarray, k: int) -> dict:
        """Compute recall metrics"""
        recalls_at_1 = []
        recalls_at_10 = []
        recalls_at_k = []
        
        for i, (res, gt) in enumerate(zip(results, ground_truth)):
            result_ids = [r[0] for r in res[:k]]
            gt_ids = set(gt[:k])
            
            # R@1
            if len(result_ids) >= 1:
                recalls_at_1.append(1.0 if result_ids[0] in gt[:1] else 0.0)
            
            # R@10
            if len(result_ids) >= 10:
                intersection = len(set(result_ids[:10]) & set(gt[:10]))
                recalls_at_10.append(intersection / 10.0)
            
            # R@k
            intersection = len(set(result_ids) & gt_ids)
            recalls_at_k.append(intersection / min(k, len(gt_ids)))
        
        return {
            'R@1': np.mean(recalls_at_1) if recalls_at_1 else 0.0,
            'R@10': np.mean(recalls_at_10) if recalls_at_10 else 0.0,
            f'R@{k}': np.mean(recalls_at_k) if recalls_at_k else 0.0
        }
    
    def _print_query_results(self, query_times: List[float], total_time: float,
                             recall_metrics: dict = None):
        """Print query results"""
        stats = self.aggregate_stats
        n = stats.total_queries
        
        print(f"\n{'='*70}")
        print(f"{'QUERY RESULTS':^70}")
        print(f"{'='*70}")
        
        print(f"\nTiming:")
        print(f"  Total time:           {total_time:.3f}s")
        print(f"  Avg time/query:       {np.mean(query_times)*1000:.3f}ms")
        print(f"  QPS:                  {n/total_time:.1f}")
        
        print(f"\nCluster Access Statistics:")
        print(f"  Total clusters accessed:    {stats.total_clusters_accessed}")
        print(f"  Avg clusters/query:         {stats.total_clusters_accessed/n:.2f}")
        print(f"  Min clusters/query:         {min(stats.clusters_accessed_per_query)}")
        print(f"  Max clusters/query:         {max(stats.clusters_accessed_per_query)}")
        
        print(f"\nCache Statistics:")
        print(f"  Total cache hits:           {stats.total_cache_hits}")
        print(f"  Avg cache hits/query:       {stats.total_cache_hits/n:.2f}")
        loads = stats.total_cache_hits + stats.total_clusters_accessed
        if loads > 0:
            print(f"  Cache hit rate:             {stats.total_cache_hits/loads:.4f}")
        
        print(f"\nHop Statistics:")
        print(f"  Total hops:                 {stats.total_hops}")
        print(f"  Avg hops/query:             {stats.total_hops/n:.2f}")
        
        print(f"\nVisit Statistics:")
        print(f"  Total nodes visited:        {stats.total_nodes_visited}")
        print(f"  Avg nodes visited/query:    {stats.total_nodes_visited/n:.2f}")
        
        if recall_metrics:
            print(f"\nRecall Metrics:")
            for metric, value in recall_metrics.items():
                print(f"  {metric}:                    {value:.4f}")
        
        print(f"{'='*70}")
    
    def print_graph_structure(self):
        """Print graph structure statistics"""
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

    def compute_neighbor_cluster_stats(self) -> dict:
        """
        For each cluster, count the number of unique *other* clusters it has at
        least one cross-cluster edge to (using base layer / level 0 edges).
        Returns summary stats and per-cluster counts.
        """
        if not self.cluster_subgraphs:
            raise RuntimeError("Clusters must be built before computing neighbor-cluster stats")

        per_cluster = {}
        counts = []

        for cluster_id, sg in self.cluster_subgraphs.items():
            neighbor_clusters: Set[int] = set()
            for cross_list in sg.cross_cluster_neighbors.values():
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
            summary = {"min": int(min(counts)), "max": int(max(counts)), "avg": float(np.mean(counts))}

        return {
            "n_clusters": int(self.n_clusters) if self.n_clusters is not None else len(self.cluster_subgraphs),
            "neighbor_clusters_per_cluster": summary,
            "per_cluster": per_cluster,
        }


def _timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def _default_log_dir() -> str:
    # Keep logs colocated with other hnsw logs by default.
    here = Path(__file__).resolve()
    return str(here.parent / "log" / "hnsw_clustered")


def run_all_datasets(num_queries: int = 100, k: int = 10, ef_search: int = 64) -> dict:
    """Run experiments on all available datasets"""
    datasets = [
        {
            'name': '1k',
            'json': 'data/1k/hnsw_32_40.json',
            'vectors': 'data/1k/laion_base.fvecs',
            'queries': 'data/1k/laion_query.fvecs',
            'gt': 'data/1k/gt.ivecs',
            'n_clusters': 31
        },
        {
            'name': '10k',
            'json': 'data/10k/hnsw_32_40.json',
            'vectors': 'data/10k/laion_base.fvecs',
            'queries': 'data/laion_query.fvecs',
            'gt': 'data/10k/gt.ivecs',
            'n_clusters': 100
        },
        {
            'name': '100k',
            'json': 'data/100k/hnsw_32_40.json',
            'vectors': 'data/100k/laion_base.fvecs',
            'queries': 'data/laion_query.fvecs',
            'gt': 'data/100k/gt.ivecs',
            'n_clusters': 316
        },
        {
            'name': '1M',
            'json': 'data/1m/hnsw_32_40.json',
            'vectors': 'data/1m/laion_base.fvecs',
            'queries': 'data/laion_query.fvecs',
            'gt': 'data/1m/gt.ivecs',
            'n_clusters': 1000
        }
    ]
    
    all_results = {}
    
    for ds in datasets:
        print(f"\n{'#'*70}")
        print(f"{'DATASET: ' + ds['name']:^70}")
        print(f"{'#'*70}")
        
        try:
            hnsw = ClusteredHNSW(n_clusters=ds['n_clusters'], metric='l2')
            
            # Load data
            hnsw.load_hnsw_json(ds['json'])
            hnsw.load_vectors(ds['vectors'])
            
            # Build clusters
            hnsw.build_clusters()
            cluster_detail_stats = hnsw.compute_cluster_detail_stats()
            hnsw._print_cluster_stats(cluster_detail_stats)

            # Neighbor cluster stats per cluster
            neighbor_cluster_stats = hnsw.compute_neighbor_cluster_stats()
            print("\n  Neighbor-Cluster Statistics (per cluster):")
            s = neighbor_cluster_stats["neighbor_clusters_per_cluster"]
            print(f"    min={s['min']}, max={s['max']}, avg={s['avg']:.2f}")
            
            # Print graph structure
            hnsw.print_graph_structure()
            
            # Load queries and ground truth
            queries, nq, d = load_fvecs(ds['queries'])
            gt = load_ivecs(ds['gt'])
            
            # Run queries
            results = hnsw.run_queries(
                queries=queries,
                k=k,
                ef_search=ef_search,
                ground_truth=gt,
                num_queries=num_queries
            )

            results["neighbor_cluster_stats"] = neighbor_cluster_stats
            results["cluster_detail_stats"] = cluster_detail_stats
            all_results[ds['name']] = results
            
        except FileNotFoundError as e:
            print(f"  Skipping {ds['name']}: {e}")
            continue
        except Exception as e:
            print(f"  Error processing {ds['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    print(f"\n{'#'*70}")
    print(f"{'SUMMARY':^70}")
    print(f"{'#'*70}")
    
    print(f"\n{'Dataset':<10} {'Clusters':<10} {'Clusters/Q':<12} {'CacheHits/Q':<12} "
          f"{'Hops/Q':<10} {'CutRatio':<10} {'R@10':<10} {'QPS':<10}")
    print(f"{'-'*84}")
    
    for name, res in all_results.items():
        r10 = res['recall']['R@10'] if res['recall'] else 0.0
        cm = res.get('cluster_metrics') or {}
        cut_ratio = cm.get('cut_ratio', 0.0)
        print(f"{name:<10} {res.get('n_clusters', 0):<10d} "
              f"{res['avg_clusters_accessed']:<12.2f} "
              f"{res['avg_cache_hits']:<12.2f} {res['avg_hops']:<10.2f} "
              f"{cut_ratio:<10.4f} "
              f"{r10:<10.4f} {res['qps']:<10.1f}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='HNSW with K-means Clustering and Subgraph Caching')
    parser.add_argument('--json', type=str, default='data/10k/hnsw_graph.json', help='HNSW JSON file')
    parser.add_argument('--vectors', type=str, default='data/10k/laion_base.fvecs', help='Base vectors')
    parser.add_argument('--queries', type=str, default='data/10k/laion_query.fvecs', help='Query vectors')
    parser.add_argument('--gt', type=str, default='data/10k/gt.ivecs', help='Ground truth')
    parser.add_argument('--n-clusters', type=int, default=None,
                        help='Number of clusters (default: sqrt(ntotal))')
    parser.add_argument('--k', type=int, default=10, help='Number of neighbors')
    parser.add_argument('--ef', type=int, default=64, help='ef_search parameter')
    parser.add_argument('--num-queries', type=int, default=100, help='Number of queries to run')
    parser.add_argument('--metric', type=str, default='l2', choices=['l2', 'ip'], help='Distance metric')
    parser.add_argument('--run-all', action='store_true', help='Run on all datasets')
    parser.add_argument('--log-dir', type=str, default=_default_log_dir(), help='Directory for JSON outputs')
    parser.add_argument('--run-name', type=str, default=None, help='Optional run name for log filenames')
    parser.add_argument('--write-neighbor-cluster-json', action='store_true',
                        help='Write neighbor-cluster stats JSON to log-dir')
    
    args = parser.parse_args()

    log_dir = _ensure_dir(args.log_dir)
    run_name = args.run_name or ("all" if args.run_all else "single")
    if args.run_all:
        all_results = run_all_datasets(num_queries=args.num_queries, k=args.k, ef_search=args.ef)
        # Save summary JSON for all datasets (includes neighbor stats)
        out_path = str(Path(log_dir) / f"hnsw_clustered_all_results_{_timestamp()}.json")
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nWrote all-datasets results JSON to {out_path}")
        return
    
    # Single dataset run
    hnsw = ClusteredHNSW(n_clusters=args.n_clusters, metric=args.metric)
    
    # Load data
    hnsw.load_hnsw_json(args.json)
    hnsw.load_vectors(args.vectors)
    
    # Build clusters
    hnsw.build_clusters()
    cluster_detail_stats = hnsw.compute_cluster_detail_stats()
    hnsw._print_cluster_stats(cluster_detail_stats)

    neighbor_cluster_stats = hnsw.compute_neighbor_cluster_stats()
    print("\nNeighbor-Cluster Statistics (per cluster):")
    s = neighbor_cluster_stats["neighbor_clusters_per_cluster"]
    print(f"  min={s['min']}, max={s['max']}, avg={s['avg']:.2f}")
    
    # Print graph structure
    hnsw.print_graph_structure()
    
    # Load queries
    queries, nq, d = load_fvecs(args.queries)
    
    # Load ground truth if available
    gt = None
    try:
        gt = load_ivecs(args.gt)
    except FileNotFoundError:
        print("Ground truth not found, skipping recall calculation")
    
    # Run queries
    results = hnsw.run_queries(
        queries=queries,
        k=args.k,
        ef_search=args.ef,
        ground_truth=gt,
        num_queries=args.num_queries
    )

    results["neighbor_cluster_stats"] = neighbor_cluster_stats
    results["cluster_detail_summary"] = cluster_detail_stats["summary"]

    timestamp = _timestamp()
    out_path = str(Path(log_dir) / f"hnsw_clustered_{run_name}_{timestamp}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote results JSON to {out_path}")

    detail_path = str(Path(log_dir) / f"hnsw_clustered_cluster_stats_{run_name}_{timestamp}.json")
    with open(detail_path, "w") as f:
        json.dump(cluster_detail_stats, f, indent=2)
    print(f"Wrote per-cluster detail stats JSON to {detail_path}")

    if args.write_neighbor_cluster_json:
        nc_path = str(Path(log_dir) / f"hnsw_clustered_neighbor_clusters_{run_name}_{timestamp}.json")
        with open(nc_path, "w") as f:
            json.dump(neighbor_cluster_stats, f, indent=2)
        print(f"Wrote neighbor-cluster stats JSON to {nc_path}")


if __name__ == '__main__':
    main()