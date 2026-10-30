#!/usr/bin/env python3
"""
IVF-HNSW Query Hop Counter - Tracks hops in centroid HNSW + cluster exhaustive search
"""

import json
import struct
import argparse
import time
import math
from heapq import heappush, heappop, heapreplace, nlargest
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass, field

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

try:
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def load_ivecs(path: str) -> List[List[int]]:
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
    return gt


# ==================== DATA STRUCTURES ====================

@dataclass
class IVFHopStats:
    """Track hop statistics for IVF-HNSW"""
    # Centroid HNSW stats (per layer)
    centroid_hops_per_layer: Dict[int, List[int]] = field(default_factory=dict)
    # Cluster search stats
    cluster_distance_computations: List[int] = field(default_factory=list)
    clusters_searched: List[int] = field(default_factory=list)
    
    def add_query_stats(self, centroid_hops: Dict[int, int], cluster_comps: int, num_clusters: int):
        for layer, hops in centroid_hops.items():
            if layer not in self.centroid_hops_per_layer:
                self.centroid_hops_per_layer[layer] = []
            self.centroid_hops_per_layer[layer].append(hops)
        self.cluster_distance_computations.append(cluster_comps)
        self.clusters_searched.append(num_clusters)


@dataclass
class QueryResult:
    """Result of a single IVF-HNSW query"""
    centroid_hops_per_layer: Dict[int, int] = field(default_factory=dict)
    cluster_distance_computations: int = 0
    clusters_searched: int = 0
    total_hops: int = 0


class IVFHNSW:
    """IVF-HNSW index with hop tracking"""
    
    def __init__(self, vectors_path: str, metric: str = "l2", nlist: int = None):
        self.metric = metric
        self.vectors = None
        self.dimension = 0
        self.ntotal = 0
        
        # IVF components
        self.nlist = nlist  # Number of clusters
        self.centroids = None  # Cluster centroids
        self.inverted_lists = {}  # cluster_id -> list of vector_ids
        self.vector_to_cluster = {}  # vector_id -> cluster_id
        
        # HNSW over centroids
        self.centroid_adjacency = {}  # layer -> {centroid_id -> [neighbors]}
        self.centroid_max_level = 0
        self.centroid_entry_point = 0
        
        # Statistics
        self.stats = IVFHopStats()
        
        # Load vectors
        self._load_vectors(vectors_path)
    
    def _load_vectors(self, vectors_path: str):
        """Load vectors from .fvecs file"""
        print(f"Loading vectors from {vectors_path}...")
        
        vectors = []
        with open(vectors_path, 'rb') as f:
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
        
        if HAS_NUMPY:
            self.vectors = np.array(vectors, dtype=np.float32)
        else:
            self.vectors = vectors
        
        self.ntotal = len(vectors)
        self.dimension = len(vectors[0]) if vectors else 0
        
        # Auto-determine nlist if not provided
        if self.nlist is None:
            self.nlist = max(4, min(256, int(math.sqrt(self.ntotal))))
        
        print(f"  Loaded: {self.ntotal} vectors of dimension {self.dimension}")
        print(f"  Using nlist={self.nlist} clusters")
    
    def _distance(self, vec1, vec2) -> float:
        """Compute distance between vectors"""
        if self.metric == "ip":
            if HAS_NUMPY and isinstance(vec1, np.ndarray):
                return -float(np.dot(vec1, vec2))
            else:
                return -sum(a * b for a, b in zip(vec1, vec2))
        else:  # L2
            if HAS_NUMPY and isinstance(vec1, np.ndarray):
                diff = vec1 - vec2
                return float(np.dot(diff, diff))
            else:
                return sum((a - b) ** 2 for a, b in zip(vec1, vec2))
    
    def build_index(self, M: int = 16, ef_construction: int = 40):
        """Build IVF-HNSW index"""
        print(f"\nBuilding IVF-HNSW index...")
        print(f"  nlist={self.nlist}, M={M}, ef_construction={ef_construction}")
        
        # Step 1: K-means clustering
        print("  Step 1: K-means clustering...")
        start_time = time.time()
        self._build_clusters()
        print(f"    Completed in {time.time() - start_time:.2f}s")
        
        # Step 2: Build HNSW over centroids
        print("  Step 2: Building HNSW over centroids...")
        start_time = time.time()
        self._build_centroid_hnsw(M=M, ef_construction=ef_construction)
        print(f"    Completed in {time.time() - start_time:.2f}s")
        
        # Print cluster statistics
        cluster_sizes = [len(self.inverted_lists[i]) for i in range(self.nlist)]
        print(f"\n  Cluster statistics:")
        print(f"    Min size: {min(cluster_sizes)}")
        print(f"    Max size: {max(cluster_sizes)}")
        print(f"    Avg size: {sum(cluster_sizes) / len(cluster_sizes):.1f}")
        
        print(f"\n  Centroid HNSW: {self.centroid_max_level + 1} layers")
    
    def _build_clusters(self):
        """Build IVF clusters using K-means"""
        if not HAS_SKLEARN:
            raise RuntimeError("sklearn required for K-means clustering")
        
        # Run K-means
        if HAS_NUMPY:
            X = self.vectors
        else:
            X = np.array(self.vectors, dtype=np.float32)
        
        kmeans = KMeans(n_clusters=self.nlist, random_state=42, n_init=10, max_iter=100)
        labels = kmeans.fit_predict(X)
        self.centroids = kmeans.cluster_centers_
        
        # Build inverted lists
        for cluster_id in range(self.nlist):
            self.inverted_lists[cluster_id] = []
        
        for vec_id, cluster_id in enumerate(labels):
            cluster_id = int(cluster_id)
            self.inverted_lists[cluster_id].append(vec_id)
            self.vector_to_cluster[vec_id] = cluster_id
    
    def _build_centroid_hnsw(self, M: int = 16, ef_construction: int = 40):
        """Build HNSW graph over cluster centroids"""
        n = self.nlist
        
        # Determine max level for each node (using HNSW level generation)
        ml = 1.0 / math.log(M)
        levels = []
        for i in range(n):
            level = int(-math.log(max(1e-10, np.random.random())) * ml)
            levels.append(min(level, 10))  # Cap at 10 levels
        
        self.centroid_max_level = max(levels)
        
        # Initialize adjacency lists
        for layer in range(self.centroid_max_level + 1):
            self.centroid_adjacency[layer] = {i: [] for i in range(n)}
        
        # Find entry point (highest level node)
        self.centroid_entry_point = levels.index(max(levels))
        
        # Insert nodes one by one
        for node_id in range(n):
            node_level = levels[node_id]
            
            if node_id == 0:
                continue
            
            # Find entry point for this node
            ep = self.centroid_entry_point
            
            # Search from top to node_level + 1
            for layer in range(self.centroid_max_level, node_level, -1):
                ep = self._greedy_search_centroid(self.centroids[node_id], layer, ep)[0]
            
            # Insert at each level from node_level down to 0
            for layer in range(min(node_level, self.centroid_max_level), -1, -1):
                # Find neighbors at this layer
                candidates = self._search_layer_centroid(
                    self.centroids[node_id], layer, ep, ef_construction
                )
                
                # Select M best neighbors
                neighbors = self._select_neighbors(node_id, candidates, M)
                
                # Add bidirectional edges
                self.centroid_adjacency[layer][node_id] = neighbors
                for neighbor in neighbors:
                    if node_id not in self.centroid_adjacency[layer][neighbor]:
                        self.centroid_adjacency[layer][neighbor].append(node_id)
                        # Prune if needed
                        if len(self.centroid_adjacency[layer][neighbor]) > M * 2:
                            self._prune_connections(neighbor, layer, M * 2)
                
                if candidates:
                    ep = candidates[0][1]  # Use closest as entry point for next layer
    
    def _select_neighbors(self, node_id: int, candidates: List[Tuple[float, int]], M: int) -> List[int]:
        """Select M best neighbors from candidates"""
        neighbors = []
        for dist, cand_id in sorted(candidates)[:M]:
            if cand_id != node_id:
                neighbors.append(cand_id)
        return neighbors[:M]
    
    def _prune_connections(self, node_id: int, layer: int, max_connections: int):
        """Prune connections to max_connections"""
        neighbors = self.centroid_adjacency[layer][node_id]
        if len(neighbors) <= max_connections:
            return
        
        # Keep closest neighbors
        distances = []
        for neighbor in neighbors:
            dist = self._distance(self.centroids[node_id], self.centroids[neighbor])
            distances.append((dist, neighbor))
        
        distances.sort()
        self.centroid_adjacency[layer][node_id] = [n for _, n in distances[:max_connections]]
    
    def _greedy_search_centroid(self, query, layer: int, start_node: int) -> Tuple[int, int]:
        """Greedy search at a single layer of centroid HNSW"""
        current = start_node
        current_dist = self._distance(self.centroids[current], query)
        hops = 0
        
        while True:
            improved = False
            neighbors = self.centroid_adjacency[layer].get(current, [])
            
            for neighbor in neighbors:
                hops += 1
                neighbor_dist = self._distance(self.centroids[neighbor], query)
                
                if neighbor_dist < current_dist:
                    current = neighbor
                    current_dist = neighbor_dist
                    improved = True
                    break
            
            if not improved:
                break
        
        return current, hops
    
    def _search_layer_centroid(self, query, layer: int, entry_node: int, ef: int) -> List[Tuple[float, int]]:
        """ef-search at a layer of centroid HNSW"""
        visited = set([entry_node])
        entry_dist = self._distance(self.centroids[entry_node], query)
        
        candidates = [(entry_dist, entry_node)]
        top_ef = [(-entry_dist, entry_node)]
        
        while candidates:
            cand_dist, cand_node = heappop(candidates)
            
            if top_ef and cand_dist > -top_ef[0][0]:
                break
            
            neighbors = self.centroid_adjacency[layer].get(cand_node, [])
            
            for neighbor in neighbors:
                if neighbor in visited:
                    continue
                
                visited.add(neighbor)
                neighbor_dist = self._distance(self.centroids[neighbor], query)
                
                if len(top_ef) < ef:
                    heappush(top_ef, (-neighbor_dist, neighbor))
                    heappush(candidates, (neighbor_dist, neighbor))
                elif neighbor_dist < -top_ef[0][0]:
                    heapreplace(top_ef, (-neighbor_dist, neighbor))
                    heappush(candidates, (neighbor_dist, neighbor))
        
        results = []
        while top_ef:
            neg_dist, node = heappop(top_ef)
            results.append((-neg_dist, node))
        results.sort()
        return results
    
    def search(self, query, k: int = 10, nprobe: int = 8, ef_search: int = 16) -> Tuple[List[Tuple[int, float]], QueryResult]:
        """
        IVF-HNSW search with hop tracking.
        
        1. Use HNSW to find top nprobe centroids
        2. Exhaustively search within those clusters
        """
        result = QueryResult()
        
        # Phase 1: Find top nprobe centroids using HNSW
        current = self.centroid_entry_point
        
        # Greedy search from top layer down to layer 1
        for layer in range(self.centroid_max_level, 0, -1):
            current, hops = self._greedy_search_centroid(query, layer, current)
            result.centroid_hops_per_layer[layer] = hops
            result.total_hops += hops
        
        # ef-search at base layer to get nprobe centroids
        ef = max(ef_search, nprobe)
        top_centroids = self._search_layer_centroid(query, 0, current, ef)
        
        # Count hops at layer 0 (approximate based on visited nodes)
        layer0_hops = len(top_centroids) + ef  # Approximation
        result.centroid_hops_per_layer[0] = layer0_hops
        result.total_hops += layer0_hops
        
        # Get top nprobe cluster IDs
        cluster_ids = [cid for _, cid in top_centroids[:nprobe]]
        result.clusters_searched = len(cluster_ids)
        
        # Phase 2: Exhaustive search within clusters
        candidates = []
        distance_computations = 0
        
        for cluster_id in cluster_ids:
            for vec_id in self.inverted_lists[cluster_id]:
                distance_computations += 1
                dist = self._distance(self.vectors[vec_id], query)
                
                if len(candidates) < k:
                    heappush(candidates, (-dist, vec_id))
                elif dist < -candidates[0][0]:
                    heapreplace(candidates, (-dist, vec_id))
        
        result.cluster_distance_computations = distance_computations
        result.total_hops += distance_computations
        
        # Extract results
        results = []
        while candidates:
            neg_dist, vec_id = heappop(candidates)
            results.append((vec_id, -neg_dist))
        results.sort(key=lambda x: x[1])
        
        return results, result
    
    def compute_recall(self, results: List[List[Tuple[int, float]]], 
                       ground_truth: List[List[int]], k: int) -> dict:
        """Compute recall@k against ground truth"""
        recalls = []
        for res, gt in zip(results, ground_truth):
            result_ids = set([r[0] for r in res[:k]])
            gt_ids = set(gt[:k])
            recall = len(result_ids & gt_ids) / min(k, len(gt_ids)) if gt_ids else 0
            recalls.append(recall)
        return {
            'avg': sum(recalls) / len(recalls) if recalls else 0,
            'min': min(recalls) if recalls else 0,
            'max': max(recalls) if recalls else 0
        }
    
    def run_queries(self, queries_path: str, num_queries: int = 100, 
                    k: int = 10, nprobe: int = 8, ef_search: int = 16,
                    gt_path: str = None):
        """Run queries and collect statistics"""
        print(f"\nLoading queries from {queries_path}...")
        queries = []
        with open(queries_path, 'rb') as f:
            while True:
                dim_bytes = f.read(4)
                if len(dim_bytes) < 4:
                    break
                dim = struct.unpack('<i', dim_bytes)[0]
                vector_bytes = f.read(dim * 4)
                if len(vector_bytes) < dim * 4:
                    break
                vector = struct.unpack('<' + 'f' * dim, vector_bytes)
                queries.append(list(vector))
        
        if HAS_NUMPY:
            queries = np.array(queries, dtype=np.float32)
        
        # Load ground truth if provided
        ground_truth = None
        if gt_path:
            print(f"  Loading ground truth from {gt_path}...")
            try:
                ground_truth = load_ivecs(gt_path)
                print(f"  Loaded {len(ground_truth)} ground truth entries")
            except Exception as e:
                print(f"  Warning: Could not load ground truth: {e}")
                ground_truth = None
        
        num_queries = min(num_queries, len(queries))
        print(f"  Running {num_queries} queries with k={k}, nprobe={nprobe}, ef_search={ef_search}")
        
        # Reset statistics
        self.stats = IVFHopStats()
        all_results = []
        
        # Run queries
        start_time = time.time()
        for i in range(num_queries):
            if (i + 1) % 100 == 0 or i == num_queries - 1:
                print(f"\r  Progress: {i + 1}/{num_queries}", end="", flush=True)
            
            query = queries[i]
            results, query_result = self.search(query, k=k, nprobe=nprobe, ef_search=ef_search)
            
            all_results.append(results)
            self.stats.add_query_stats(
                query_result.centroid_hops_per_layer,
                query_result.cluster_distance_computations,
                query_result.clusters_searched
            )
        
        elapsed = time.time() - start_time
        print(f"\n  Completed in {elapsed:.2f}s ({num_queries/elapsed:.1f} QPS)")
        
        # Store timing info for statistics
        self.query_time = elapsed
        self.num_queries_run = num_queries
        
        # Compute recall if ground truth is available
        self.recall_metrics = None
        if ground_truth:
            self.recall_metrics = self.compute_recall(all_results, ground_truth[:num_queries], k)
            print(f"  Recall@{k}: {self.recall_metrics['avg']:.4f}")
        
        return num_queries
    
    def print_statistics(self, dataset_name: str = ""):
        """Print hop statistics"""
        print(f"\n{'='*80}")
        print(f"IVF-HNSW HOP STATISTICS: {dataset_name}" if dataset_name else "IVF-HNSW HOP STATISTICS")
        print(f"{'='*80}")
        print(f"Dataset size: {self.ntotal:,} vectors")
        print(f"Number of clusters (nlist): {self.nlist}")
        print(f"Centroid HNSW layers: {self.centroid_max_level + 1}")
        print(f"Centroid HNSW entry point: {self.centroid_entry_point}")
        
        # Centroid HNSW statistics
        print(f"\n{'-'*80}")
        print(f"CENTROID HNSW SEARCH (finding top nprobe clusters)")
        print(f"{'-'*80}")
        print(f"{'Layer':<8} {'Avg Hops':<12} {'Min Hops':<10} {'Max Hops':<10} {'Total Hops':<12}")
        print(f"{'-'*80}")
        
        total_centroid_hops = 0
        num_queries = len(self.stats.cluster_distance_computations)
        
        for layer in range(self.centroid_max_level, -1, -1):
            if layer in self.stats.centroid_hops_per_layer:
                hops_list = self.stats.centroid_hops_per_layer[layer]
                avg_hops = sum(hops_list) / len(hops_list) if hops_list else 0
                min_hops = min(hops_list) if hops_list else 0
                max_hops = max(hops_list) if hops_list else 0
                total_hops = sum(hops_list)
                total_centroid_hops += total_hops
                print(f"L{layer:<7} {avg_hops:<12.2f} {min_hops:<10} {max_hops:<10} {total_hops:<12,}")
        
        avg_centroid_total = total_centroid_hops / num_queries if num_queries > 0 else 0
        print(f"{'-'*80}")
        print(f"{'Subtotal':<8} {avg_centroid_total:<12.2f} {'':<10} {'':<10} {total_centroid_hops:<12,}")
        
        # Cluster search statistics
        print(f"\n{'-'*80}")
        print(f"CLUSTER EXHAUSTIVE SEARCH")
        print(f"{'-'*80}")
        
        cluster_comps = self.stats.cluster_distance_computations
        clusters_searched = self.stats.clusters_searched
        
        avg_comps = sum(cluster_comps) / len(cluster_comps) if cluster_comps else 0
        min_comps = min(cluster_comps) if cluster_comps else 0
        max_comps = max(cluster_comps) if cluster_comps else 0
        total_comps = sum(cluster_comps)
        
        avg_clusters = sum(clusters_searched) / len(clusters_searched) if clusters_searched else 0
        
        print(f"Avg clusters searched per query: {avg_clusters:.1f}")
        print(f"Avg distance computations/query: {avg_comps:.2f}")
        print(f"Min distance computations: {min_comps}")
        print(f"Max distance computations: {max_comps}")
        print(f"Total distance computations: {total_comps:,}")
        
        # Total statistics
        print(f"\n{'-'*80}")
        print(f"TOTAL STATISTICS")
        print(f"{'-'*80}")
        
        total_hops = total_centroid_hops + total_comps
        avg_total = total_hops / num_queries if num_queries > 0 else 0
        
        print(f"{'Component':<30} {'Avg/Query':<15} {'Total':<15}")
        print(f"{'-'*80}")
        print(f"{'Centroid HNSW hops':<30} {avg_centroid_total:<15.2f} {total_centroid_hops:<15,}")
        print(f"{'Cluster search (dist comps)':<30} {avg_comps:<15.2f} {total_comps:<15,}")
        print(f"{'-'*80}")
        print(f"{'TOTAL':<30} {avg_total:<15.2f} {total_hops:<15,}")
        
        # ==================== RECALL & SPEED STATISTICS ====================
        print(f"\n{'-'*80}")
        print(f"RECALL & SPEED STATISTICS")
        print(f"{'-'*80}")
        
        if hasattr(self, 'recall_metrics') and self.recall_metrics:
            print(f"  Recall@k:")
            print(f"    Average: {self.recall_metrics['avg']:.4f}")
            print(f"    Min:     {self.recall_metrics['min']:.4f}")
            print(f"    Max:     {self.recall_metrics['max']:.4f}")
        else:
            print(f"  Recall: Not computed (no ground truth provided)")
        
        if hasattr(self, 'query_time') and hasattr(self, 'num_queries_run'):
            qps = self.num_queries_run / self.query_time if self.query_time > 0 else 0
            print(f"  Speed:")
            print(f"    Total time: {self.query_time:.2f}s")
            print(f"    QPS: {qps:.2f}")
        
        print(f"{'='*80}\n")
        
        result = {
            'dataset_size': self.ntotal,
            'nlist': self.nlist,
            'centroid_layers': self.centroid_max_level + 1,
            'num_queries': num_queries,
            'avg_centroid_hops': avg_centroid_total,
            'avg_cluster_comps': avg_comps,
            'avg_total_hops': avg_total,
        }
        
        # Add recall metrics if available
        if hasattr(self, 'recall_metrics') and self.recall_metrics:
            result['recall_avg'] = self.recall_metrics['avg']
            result['recall_min'] = self.recall_metrics['min']
            result['recall_max'] = self.recall_metrics['max']
        
        # Add speed metrics if available
        if hasattr(self, 'query_time') and hasattr(self, 'num_queries_run'):
            result['query_time'] = self.query_time
            result['qps'] = self.num_queries_run / self.query_time if self.query_time > 0 else 0
        
        return result


def main():
    parser = argparse.ArgumentParser(description='IVF-HNSW Query Hop Counter')
    parser.add_argument('--vectors', type=str, required=True, help='Path to base vectors (.fvecs)')
    parser.add_argument('--queries', type=str, required=True, help='Path to query vectors (.fvecs)')
    parser.add_argument('--gt', type=str, default=None, help='Ground truth .ivecs file')
    parser.add_argument('--num-queries', type=int, default=100, help='Number of queries to run')
    parser.add_argument('--k', type=int, default=10, help='Number of nearest neighbors')
    parser.add_argument('--nlist', type=int, default=None, help='Number of clusters (auto if not specified)')
    parser.add_argument('--nprobe', type=int, default=8, help='Number of clusters to search')
    parser.add_argument('--ef-search', type=int, default=16, help='ef_search for centroid HNSW')
    parser.add_argument('--M', type=int, default=16, help='HNSW M parameter')
    parser.add_argument('--metric', type=str, default='l2', choices=['ip', 'l2'], help='Distance metric')
    parser.add_argument('--name', type=str, default='', help='Dataset name for output')
    
    args = parser.parse_args()
    
    # Create IVF-HNSW index
    index = IVFHNSW(
        vectors_path=args.vectors,
        metric=args.metric,
        nlist=args.nlist
    )
    
    # Build index
    index.build_index(M=args.M)
    
    # Run queries
    index.run_queries(
        queries_path=args.queries,
        num_queries=args.num_queries,
        k=args.k,
        nprobe=args.nprobe,
        ef_search=args.ef_search,
        gt_path=args.gt
    )
    
    # Print statistics
    index.print_statistics(dataset_name=args.name)


def run_all_datasets():
    """Run all datasets"""
    datasets = [
        {
            'name': '1k',
            'vectors': 'data/1k/laion_base.fvecs',
            'queries': 'data/1k/laion_query.fvecs',
            'gt': 'data/1k/gt.ivecs',
            'nlist': 32,
        },
        {
            'name': '10k',
            'vectors': 'data/10k/laion_base.fvecs',
            'queries': 'data/laion_query.fvecs',
            'gt': 'data/10k/gt.ivecs',
            'nlist': 100,
        },
        {
            'name': '100k',
            'vectors': 'data/100k/laion_base.fvecs',
            'queries': 'data/laion_query.fvecs',
            'gt': 'data/100k/gt.ivecs',
            'nlist': 316,
        },
        {
            'name': '1M',
            'vectors': 'data/1m/laion_base.fvecs',
            'queries': 'data/laion_query.fvecs',
            'gt': 'data/1m/gt.ivecs',
            'nlist': 1000,
        },
    ]
    
    all_results = {}
    
    for ds in datasets:
        print(f"\n{'#'*80}")
        print(f"# Processing Dataset: {ds['name']}")
        print(f"{'#'*80}")
        
        try:
            index = IVFHNSW(
                vectors_path=ds['vectors'],
                metric='l2',
                nlist=ds['nlist']
            )
            
            index.build_index(M=16)
            
            index.run_queries(
                queries_path=ds['queries'],
                num_queries=100,
                k=10,
                nprobe=8,
                ef_search=16,
                gt_path=ds.get('gt')
            )
            
            results = index.print_statistics(dataset_name=ds['name'])
            all_results[ds['name']] = results
            
        except Exception as e:
            print(f"Error processing {ds['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print comparison table
    print(f"\n{'='*120}")
    print(f"{'IVF-HNSW COMPARISON SUMMARY':^120}")
    print(f"{'='*120}")
    print(f"{'Dataset':<10} {'Size':>12} {'nlist':>8} {'Centroid Hops':>15} {'Cluster Comps':>15} {'Total':>12} {'Recall@10':>12} {'QPS':>10}")
    print(f"{'-'*120}")
    
    for name, res in all_results.items():
        recall = res.get('recall_avg', 0)
        qps = res.get('qps', 0)
        print(f"{name:<10} {res['dataset_size']:>12,} {res['nlist']:>8} {res['avg_centroid_hops']:>15.2f} {res['avg_cluster_comps']:>15.2f} {res['avg_total_hops']:>12.2f} {recall:>12.4f} {qps:>10.2f}")
    
    print(f"{'='*120}\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        run_all_datasets()
    else:
        main()