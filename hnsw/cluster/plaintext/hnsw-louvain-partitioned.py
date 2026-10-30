#!/usr/bin/env python3
"""
HNSW with Louvain Community Partitioning (L0 only) and Subgraph Caching

Key property: query algorithm matches `hnsw-clustered.py`.
- Upper layers (L>0): standard greedy HNSW descent on the original graph.
- Base layer (L0): standard ef-search, but neighbor access is routed through a
  per-community cached subgraph built from Louvain communities on L0.

This uses Louvain ONLY for detecting communities / partitioning on L0.
It does NOT preselect communities (no nprobe) and does NOT forbid cross-community
edges during traversal (cross edges are preserved as in `hnsw-clustered.py`).
"""

import argparse
import json
import time
import struct
from dataclasses import dataclass, field
from heapq import heappush, heappop, heapreplace
from typing import Dict, List, Tuple, Set, Optional

import numpy as np

try:
    import networkx as nx
    from networkx.algorithms.community import louvain_communities
    _HAS_NETWORKX = True
except Exception:
    _HAS_NETWORKX = False
    nx = None  # type: ignore
    louvain_communities = None  # type: ignore


# ----------------------------
# IO helpers
# ----------------------------

def load_fvecs(path: str) -> Tuple[np.ndarray, int, int]:
    """Load vectors from .fvecs format"""
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
    """Load ground truth from .ivecs format"""
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


# ----------------------------
# Data structures
# ----------------------------

@dataclass
class CommunitySubgraph:
    """Represents a Louvain community subgraph from the base layer"""
    community_id: int
    nodes: Set[int] = field(default_factory=set)
    # node_id -> list of neighbor_ids (intra-community only)
    intra_neighbors: Dict[int, List[int]] = field(default_factory=dict)
    # node_id -> list of (neighbor_id, neighbor_community_id) for cross-community edges
    cross_neighbors: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict)


@dataclass
class QueryStats:
    """Statistics for a single query"""
    communities_accessed: int = 0
    community_cache_hits: int = 0
    total_hops: int = 0
    hops_per_layer: Dict[int, int] = field(default_factory=dict)
    nodes_visited: int = 0


@dataclass
class AggregateStats:
    total_queries: int = 0
    total_communities_accessed: int = 0
    total_cache_hits: int = 0
    total_hops: int = 0
    communities_accessed_per_query: List[int] = field(default_factory=list)
    cache_hits_per_query: List[int] = field(default_factory=list)


# ----------------------------
# Core implementation
# ----------------------------

class LouvainPartitionedHNSW:
    """
    HNSW where L0 is partitioned by Louvain communities and accessed via cached
    per-community subgraphs. Query algorithm mirrors `hnsw-clustered.py`.
    """

    def __init__(self, metric: str = "l2", resolution: float = 1.0, seed: int = 42):
        self.metric = metric
        self.resolution = float(resolution)
        self.seed = int(seed)

        # HNSW structure
        self.max_level: int = 0
        self.entry_point: Optional[int] = None
        self.ntotal: int = 0
        self.vectors: Dict[int, np.ndarray] = {}
        self.level_graphs: Dict[int, Dict[int, List[int]]] = {}

        # Louvain community structures (L0 only)
        self.node_to_community: Dict[int, int] = {}
        self.community_subgraphs: Dict[int, CommunitySubgraph] = {}
        self.num_communities: int = 0

        # Stats
        self.aggregate_stats = AggregateStats()

    # ---- graph/vector loading ----

    def load_hnsw_json(self, json_path: str) -> dict:
        """Load HNSW graph from JSON file"""
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
        """Load base vectors"""
        print(f"Loading vectors from {vectors_path}...")
        X, n, d = load_fvecs(vectors_path)
        for i in range(n):
            self.vectors[i] = X[i]
        print(f"  Loaded: {n} vectors of dimension {d}")

    # ---- Louvain partitioning on L0 ----

    def build_communities(self) -> None:
        """Run Louvain on base layer (L0) and build cached community subgraphs."""
        if not _HAS_NETWORKX:
            raise RuntimeError(
                "networkx is required for Louvain partitioning. "
                "Install it (e.g. `pip install -r requirements.txt`) and retry."
            )
        if not self.vectors:
            raise RuntimeError("Vectors must be loaded before building communities")

        print(f"\nBuilding Louvain communities on L0 (resolution={self.resolution})...")
        start_time = time.time()

        base_layer = self.level_graphs.get(0, {})

        # Build an undirected graph for Louvain from L0 adjacency
        G = nx.Graph()
        G.add_nodes_from(range(self.ntotal))
        for u, neighs in base_layer.items():
            for v in neighs:
                if u < v:
                    G.add_edge(u, v)

        print(f"  L0 graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        communities_list = louvain_communities(G, resolution=self.resolution, seed=self.seed)

        self.node_to_community.clear()
        self.community_subgraphs.clear()

        for comm_id, nodes in enumerate(communities_list):
            sg = CommunitySubgraph(community_id=comm_id)
            sg.nodes = set(int(x) for x in nodes)
            self.community_subgraphs[comm_id] = sg
            for node_id in sg.nodes:
                self.node_to_community[int(node_id)] = comm_id

        self.num_communities = len(self.community_subgraphs)
        sizes = np.array([len(sg.nodes) for sg in self.community_subgraphs.values()], dtype=np.int64)
        self._print_community_size_stats(sizes)

        # Build intra/cross neighbor lists from L0 edges
        self._build_community_subgraphs_from_L0()

        elapsed = time.time() - start_time
        print(f"  Louvain partitioning completed in {elapsed:.2f}s")

    def _print_community_size_stats(self, sizes: np.ndarray) -> None:
        """Print additional community size/imbalance diagnostics."""
        if sizes.size == 0:
            print("  Found 0 communities (unexpected)")
            return

        sizes_sorted = np.sort(sizes)

        def pct(p: float) -> float:
            return float(np.percentile(sizes_sorted, p))

        mean = float(sizes_sorted.mean())
        median = pct(50)
        p90 = pct(90)
        p95 = pct(95)
        p99 = pct(99)
        min_sz = int(sizes_sorted[0])
        max_sz = int(sizes_sorted[-1])
        # Population std (numpy default); sample std when n>1 for intuition
        std_pop = float(np.std(sizes_sorted))
        std_sample = float(np.std(sizes_sorted, ddof=1)) if sizes_sorted.size > 1 else 0.0

        # Gini coefficient (0=perfectly equal, 1=highly imbalanced)
        # Based on mean absolute difference: G = sum_i sum_j |xi-xj| / (2 n^2 mean)
        n = float(sizes_sorted.size)
        if mean > 0:
            idx = np.arange(1, sizes_sorted.size + 1, dtype=np.float64)
            gini = float((2.0 * np.sum(idx * sizes_sorted) / (n * np.sum(sizes_sorted))) - ((n + 1.0) / n))
        else:
            gini = 0.0

        top10 = list(map(int, sizes_sorted[-10:][::-1])) if sizes_sorted.size >= 10 else list(map(int, sizes_sorted[::-1]))

        print(f"  Found {self.num_communities} communities")
        print(
            f"    Size stats: min={min_sz}, max={max_sz}, mean={mean:.1f}, median={median:.1f}, "
            f"std={std_pop:.1f}, std_sample={std_sample:.1f}"
        )
        print(f"    Percentiles: p90={p90:.1f}, p95={p95:.1f}, p99={p99:.1f}")
        if median > 0:
            print(f"    Imbalance: max/median={max_sz/median:.2f}, gini={gini:.3f}")
        else:
            print(f"    Imbalance: gini={gini:.3f}")
        print(f"    Top communities (sizes): {top10}")

    def _build_community_subgraphs_from_L0(self) -> None:
        """Build intra-community and cross-community edges from L0."""
        print("  Building community subgraphs...")
        base_layer = self.level_graphs.get(0, {})

        # Ensure all nodes exist (even isolated)
        for node_id in range(self.ntotal):
            comm = self.node_to_community.get(node_id)
            if comm is None:
                # Should not happen since we add all nodes to the NX graph
                continue
            sg = self.community_subgraphs[comm]
            sg.intra_neighbors.setdefault(node_id, [])
            sg.cross_neighbors.setdefault(node_id, [])

        intra_total = 0
        cross_total = 0

        for node_id, neighbors in base_layer.items():
            node_comm = self.node_to_community[node_id]
            sg = self.community_subgraphs[node_comm]

            intra = []
            cross = []
            for nb in neighbors:
                nb_comm = self.node_to_community[nb]
                if nb_comm == node_comm:
                    intra.append(nb)
                else:
                    cross.append((nb, nb_comm))

            sg.intra_neighbors[node_id] = intra
            sg.cross_neighbors[node_id] = cross
            intra_total += len(intra)
            cross_total += len(cross)

        print(f"    Intra-community edges: {intra_total}")
        print(f"    Cross-community edges: {cross_total}")

    # ---- distance + search ----

    def _distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        if self.metric == "l2":
            diff = vec1 - vec2
            return float(np.dot(diff, diff))
        if self.metric == "ip":
            return float(-np.dot(vec1, vec2))
        raise ValueError(f"Unknown metric: {self.metric}")

    def search(self, query: np.ndarray, k: int = 10, ef_search: int = 64) -> Tuple[List[Tuple[int, float]], QueryStats]:
        """
        Query algorithm matches `hnsw-clustered.py`:
        - Upper layers: greedy search without partitioning.
        - L0: ef-search with community subgraph caching.
        """
        stats = QueryStats()

        if self.entry_point is None:
            return [], stats

        # Per-query cache of community subgraphs
        community_cache: Dict[int, CommunitySubgraph] = {}
        current_closest = int(self.entry_point)

        # Phase 1: Greedy descent on upper layers (no partitioning)
        for level in range(self.max_level, 0, -1):
            current_closest, level_hops = self._greedy_search_layer(current_closest, query, level)
            stats.hops_per_layer[level] = level_hops
            stats.total_hops += level_hops

        # Phase 2: ef-search on L0 using cached community neighbor access
        results, base_stats = self._ef_search_with_caching(current_closest, query, ef_search, k, community_cache)
        stats.hops_per_layer[0] = base_stats["hops"]
        stats.total_hops += base_stats["hops"]
        stats.communities_accessed = base_stats["communities_accessed"]
        stats.community_cache_hits = base_stats["cache_hits"]
        stats.nodes_visited = base_stats["nodes_visited"]

        return results[:k], stats

    def _greedy_search_layer(self, entry_node: int, query: np.ndarray, level: int) -> Tuple[int, int]:
        """Greedy search at an upper layer (no partitioning), same behavior as `hnsw-clustered.py`."""
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
        community_cache: Dict[int, CommunitySubgraph],
    ) -> Tuple[List[Tuple[int, float]], dict]:
        """
        ef-search at L0 with community subgraph caching.
        Mirrors `ClusteredHNSW._ef_search_with_caching`.
        """
        visited_nodes: Set[int] = set()
        communities_accessed: Set[int] = set()
        cache_hits = 0
        hops = 0

        candidates: List[Tuple[float, int]] = []        # min-heap
        dynamic_candidates: List[Tuple[float, int]] = []  # max-heap by (-dist, id)

        if entry_node not in self.vectors:
            return [], {"hops": 0, "communities_accessed": 0, "cache_hits": 0, "nodes_visited": 0}

        entry_dist = self._distance(self.vectors[entry_node], query)
        heappush(candidates, (entry_dist, entry_node))
        heappush(dynamic_candidates, (-entry_dist, entry_node))
        visited_nodes.add(entry_node)

        # Cache the entry node's community
        entry_comm = self.node_to_community[entry_node]
        if entry_comm not in community_cache:
            community_cache[entry_comm] = self.community_subgraphs[entry_comm]
            communities_accessed.add(entry_comm)

        while candidates:
            current_dist, current_node = heappop(candidates)

            # Early termination (same logic as clustered)
            if len(dynamic_candidates) >= ef:
                worst_dist = -dynamic_candidates[0][0]
                if current_dist > worst_dist:
                    break

            neighbors, was_cache_hit = self._get_cached_neighbors(current_node, community_cache, communities_accessed)
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
            "communities_accessed": len(communities_accessed),
            "cache_hits": cache_hits,
            "nodes_visited": len(visited_nodes),
        }

    def _get_cached_neighbors(
        self,
        node_id: int,
        community_cache: Dict[int, CommunitySubgraph],
        communities_accessed: Set[int],
    ) -> Tuple[List[int], bool]:
        """
        Return neighbors for node_id from cached community subgraph.
        Includes both intra-community and cross-community edges (preserving L0 connectivity).
        Mirrors `ClusteredHNSW._get_cached_neighbors`.
        """
        comm_id = self.node_to_community[node_id]
        was_cache_hit = comm_id in community_cache

        if not was_cache_hit:
            community_cache[comm_id] = self.community_subgraphs[comm_id]
            communities_accessed.add(comm_id)

        sg = community_cache[comm_id]
        neighbors: List[int] = []
        neighbors.extend(sg.intra_neighbors.get(node_id, []))
        for nb_id, _ in sg.cross_neighbors.get(node_id, []):
            neighbors.append(nb_id)
        return neighbors, was_cache_hit

    # ---- benchmarking helpers ----

    def run_queries(
        self,
        queries: np.ndarray,
        k: int = 10,
        ef_search: int = 64,
        ground_truth: Optional[np.ndarray] = None,
        num_queries: Optional[int] = None,
    ) -> dict:
        if num_queries is not None:
            queries = queries[:num_queries]
            if ground_truth is not None:
                ground_truth = ground_truth[:num_queries]

        print(f"\n{'='*70}")
        print(f"Running {len(queries)} queries (k={k}, ef={ef_search})")
        print(f"{'='*70}")

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

        self._print_query_results(query_times, total_time, recall_metrics)

        return {
            "num_queries": len(queries),
            "total_time": total_time,
            "avg_time_per_query": float(np.mean(query_times)) if query_times else 0.0,
            "qps": (len(queries) / total_time) if total_time > 0 else 0.0,
            "avg_communities_accessed": float(np.mean([s.communities_accessed for s in all_stats])) if all_stats else 0.0,
            "avg_cache_hits": float(np.mean([s.community_cache_hits for s in all_stats])) if all_stats else 0.0,
            "avg_hops": float(np.mean([s.total_hops for s in all_stats])) if all_stats else 0.0,
            "recall": recall_metrics,
            "num_communities": self.num_communities,
        }

    def _compute_aggregate_stats(self, all_stats: List[QueryStats]) -> None:
        self.aggregate_stats = AggregateStats(
            total_queries=len(all_stats),
            total_communities_accessed=sum(s.communities_accessed for s in all_stats),
            total_cache_hits=sum(s.community_cache_hits for s in all_stats),
            total_hops=sum(s.total_hops for s in all_stats),
            communities_accessed_per_query=[s.communities_accessed for s in all_stats],
            cache_hits_per_query=[s.community_cache_hits for s in all_stats],
        )

    def _compute_recall(self, results: List[List[Tuple[int, float]]], ground_truth: np.ndarray, k: int) -> dict:
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

    def _print_query_results(self, query_times: List[float], total_time: float, recall_metrics: Optional[dict] = None) -> None:
        stats = self.aggregate_stats
        n = stats.total_queries if stats.total_queries > 0 else 1

        print(f"\n{'='*70}")
        print(f"{'QUERY RESULTS':^70}")
        print(f"{'='*70}")

        print("\nTiming:")
        print(f"  Total time:           {total_time:.3f}s")
        print(f"  Avg time/query:       {np.mean(query_times)*1000:.3f}ms")
        print(f"  QPS:                  {stats.total_queries/total_time:.1f}" if total_time > 0 else "  QPS:                  N/A")

        print("\nCommunity Access Statistics:")
        print(f"  Total communities accessed: {stats.total_communities_accessed}")
        print(f"  Avg communities/query:      {stats.total_communities_accessed/n:.2f}")
        if stats.communities_accessed_per_query:
            print(f"  Min communities/query:      {min(stats.communities_accessed_per_query)}")
            print(f"  Max communities/query:      {max(stats.communities_accessed_per_query)}")

        print("\nCache Statistics:")
        print(f"  Total cache hits:           {stats.total_cache_hits}")
        print(f"  Avg cache hits/query:       {stats.total_cache_hits/n:.2f}")

        print("\nHop Statistics:")
        print(f"  Total hops:                 {stats.total_hops}")
        print(f"  Avg hops/query:             {stats.total_hops/n:.2f}")

        if recall_metrics:
            print("\nRecall Metrics:")
            for metric, value in recall_metrics.items():
                print(f"  {metric}:                    {value:.4f}")

        print(f"{'='*70}")


def run_all_datasets(resolution: float = 1.0, ef: int = 64, k: int = 10, num_queries: int = 100) -> dict:
    datasets = [
        {
            "name": "1k",
            "json": "data/1k/hnsw_32_40.json",
            "vectors": "data/1k/laion_base.fvecs",
            "queries": "data/1k/laion_query.fvecs",
            "gt": "data/1k/gt.ivecs",
        },
        {
            "name": "10k",
            "json": "data/10k/hnsw_32_40.json",
            "vectors": "data/10k/laion_base.fvecs",
            "queries": "data/laion_query.fvecs",
            "gt": "data/10k/gt.ivecs",
        },
        {
            "name": "100k",
            "json": "data/100k/hnsw_32_40.json",
            "vectors": "data/100k/laion_base.fvecs",
            "queries": "data/laion_query.fvecs",
            "gt": "data/100k/gt.ivecs",
        },
        {
            "name": "1M",
            "json": "data/1m/hnsw_32_40.json",
            "vectors": "data/1m/laion_base.fvecs",
            "queries": "data/laion_query.fvecs",
            "gt": "data/1m/gt.ivecs",
        },
    ]

    all_results = {}

    for ds in datasets:
        print(f"\n{'#'*70}")
        print(f"{'DATASET: ' + ds['name']:^70}")
        print(f"{'#'*70}")

        hnsw = LouvainPartitionedHNSW(metric="l2", resolution=resolution, seed=42)
        hnsw.load_hnsw_json(ds["json"])
        hnsw.load_vectors(ds["vectors"])
        hnsw.build_communities()

        queries, _, _ = load_fvecs(ds["queries"])
        gt = load_ivecs(ds["gt"])

        res = hnsw.run_queries(
            queries=queries,
            k=k,
            ef_search=ef,
            ground_truth=gt,
            num_queries=num_queries,
        )
        all_results[ds["name"]] = res

    print(f"\n{'#'*70}")
    print(f"{'SUMMARY':^70}")
    print(f"{'#'*70}")

    print(f"\n{'Dataset':<10} {'Communities':<12} {'Comms/Q':<10} {'CacheHits/Q':<12} {'Hops/Q':<10} {'R@10':<10} {'QPS':<10}")
    print(f"{'-'*70}")
    for name, res in all_results.items():
        r10 = (res.get("recall") or {}).get("R@10", 0.0)
        print(
            f"{name:<10} {res.get('num_communities', 0):<12d} "
            f"{res.get('avg_communities_accessed', 0.0):<10.2f} "
            f"{res.get('avg_cache_hits', 0.0):<12.2f} "
            f"{res.get('avg_hops', 0.0):<10.2f} "
            f"{r10:<10.4f} "
            f"{res.get('qps', 0.0):<10.1f}"
        )

    return all_results


def main():
    ap = argparse.ArgumentParser(description="HNSW with Louvain partitioning on L0 and clustered-style query algorithm")
    ap.add_argument("--json", type=str, required=True, help="HNSW JSON file")
    ap.add_argument("--vectors", type=str, required=True, help="Base vectors (.fvecs)")
    ap.add_argument("--queries", type=str, required=True, help="Query vectors (.fvecs)")
    ap.add_argument("--gt", type=str, default="", help="Ground truth (.ivecs). If omitted, recall is skipped.")
    ap.add_argument("--resolution", type=float, default=1.0, help="Louvain resolution parameter")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for Louvain")
    ap.add_argument("--k", type=int, default=10, help="Number of neighbors")
    ap.add_argument("--ef", type=int, default=64, help="ef_search parameter")
    ap.add_argument("--num-queries", type=int, default=100, help="Number of queries to run")
    ap.add_argument("--metric", type=str, default="l2", choices=["l2", "ip"], help="Distance metric")
    ap.add_argument("--run-all", action="store_true", help="Run on all datasets under data/")

    args = ap.parse_args()

    if args.run_all:
        run_all_datasets(resolution=args.resolution, ef=args.ef, k=args.k, num_queries=args.num_queries)
        return

    hnsw = LouvainPartitionedHNSW(metric=args.metric, resolution=args.resolution, seed=args.seed)
    hnsw.load_hnsw_json(args.json)
    hnsw.load_vectors(args.vectors)
    hnsw.build_communities()

    queries, _, _ = load_fvecs(args.queries)
    gt = None
    if args.gt:
        try:
            gt = load_ivecs(args.gt)
        except FileNotFoundError:
            print("Ground truth not found, skipping recall calculation")
            gt = None

    hnsw.run_queries(queries=queries, k=args.k, ef_search=args.ef, ground_truth=gt, num_queries=args.num_queries)


if __name__ == "__main__":
    main()

