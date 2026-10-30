#!/usr/bin/env python3
"""
DiskANN-Style Partitioned Search with Full Vamana Architecture
- K-means clustering with overlapping assignments
- Vamana graph over cluster centroids
- Vamana subgraph INSIDE each cluster
- Inter-cluster boundary connections
"""

import json
import struct
import argparse
import time
import math
import random
from heapq import heappush, heappop, heapreplace, nlargest
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass, field
from collections import defaultdict

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
class DiskANNStats:
    """Comprehensive statistics for DiskANN search"""
    # Centroid graph search
    centroid_search_hops: List[int] = field(default_factory=list)
    
    # Cluster access (key metric!)
    clusters_accessed_per_query: List[int] = field(default_factory=list)
    
    # Intra-cluster Vamana search (per layer within cluster graph)
    intra_cluster_hops: List[int] = field(default_factory=list)
    
    # Inter-cluster traversals (crossing cluster boundaries)
    inter_cluster_hops: List[int] = field(default_factory=list)
    
    # Vectors visited
    vectors_visited_per_query: List[int] = field(default_factory=list)
    
    # Total
    total_hops_per_query: List[int] = field(default_factory=list)


@dataclass
class ClusterGraph:
    """Vamana graph for a single cluster"""
    cluster_id: int
    node_ids: List[int]  # Vector IDs in this cluster
    local_to_global: Dict[int, int]  # Local index -> global vector ID
    global_to_local: Dict[int, int]  # Global vector ID -> local index
    adjacency: Dict[int, List[int]]  # Local ID -> [local neighbor IDs]
    entry_point: int  # Local entry point
    
    # Boundary nodes (for inter-cluster connections)
    boundary_nodes: Set[int]  # Local IDs of boundary nodes
    boundary_connections: Dict[int, List[Tuple[int, int]]]  # local_id -> [(other_cluster, other_local_id)]


class DiskANNIndex:
    """Full DiskANN-style index with Vamana everywhere"""
    
    def __init__(self, vectors_path: str, metric: str = "l2",
                 n_clusters: int = None, expansion_factor: float = 1.3,
                 R_centroid: int = 32, R_cluster: int = 32,
                 alpha: float = 1.2, boundary_fraction: float = 0.2,
                 build_L_centroid: int = 200, build_L_cluster: int = 100):
        """
        Args:
            vectors_path: Path to vectors
            metric: Distance metric (l2 or ip)
            n_clusters: Number of clusters
            expansion_factor: For overlapping cluster assignment
            R_centroid: Out-degree for centroid Vamana
            R_cluster: Out-degree for intra-cluster Vamana
            alpha: α-pruning parameter
            boundary_fraction: Fraction of cluster nodes near boundary
        """
        self.metric = metric
        self.expansion_factor = expansion_factor
        self.R_centroid = R_centroid
        self.R_cluster = R_cluster
        self.alpha = alpha
        self.boundary_fraction = boundary_fraction
        self.build_L_centroid = build_L_centroid
        self.build_L_cluster = build_L_cluster
        
        # Vectors
        self.vectors = None
        self.ntotal = 0
        self.dimension = 0
        
        # Clustering
        self.n_clusters = n_clusters
        self.centroids = None
        self.vector_to_clusters = {}  # vector_id -> list of cluster_ids
        
        # Centroid Vamana graph
        self.centroid_graph = {}
        self.centroid_entry = 0
        
        # Cluster Vamana graphs
        self.cluster_graphs: Dict[int, ClusterGraph] = {}
        
        # Statistics
        self.stats = DiskANNStats()
        
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
        
        if self.n_clusters is None:
            self.n_clusters = max(4, min(500, int(math.sqrt(self.ntotal))))
        
        print(f"  Loaded: {self.ntotal} vectors of dimension {self.dimension}")
        print(f"  Target clusters: {self.n_clusters}")
    
    def _distance(self, vec1, vec2) -> float:
        """Compute distance"""
        if self.metric == "ip":
            if HAS_NUMPY and isinstance(vec1, np.ndarray):
                return -float(np.dot(vec1, vec2))
            else:
                return -sum(a * b for a, b in zip(vec1, vec2))
        else:
            if HAS_NUMPY and isinstance(vec1, np.ndarray):
                diff = vec1 - vec2
                return float(np.dot(diff, diff))
            else:
                return sum((a - b) ** 2 for a, b in zip(vec1, vec2))
    
    def _distance_by_id(self, id1: int, id2: int) -> float:
        """Compute distance between vectors by their IDs"""
        return self._distance(self.vectors[id1], self.vectors[id2])
    
    def build_index(self):
        """Build complete DiskANN index"""
        print(f"\n{'='*70}")
        print(f"Building DiskANN Index")
        print(f"{'='*70}")
        print(f"  n_clusters: {self.n_clusters}")
        print(f"  expansion_factor: {self.expansion_factor}")
        print(f"  R_centroid: {self.R_centroid}")
        print(f"  R_cluster: {self.R_cluster}")
        print(f"  α: {self.alpha}")
        print(f"  boundary_fraction: {self.boundary_fraction}")
        print(f"  build_L_centroid: {self.build_L_centroid}")
        print(f"  build_L_cluster: {self.build_L_cluster}")
        
        # Step 1: K-means clustering
        print(f"\n{'─'*70}")
        print("Step 1: K-means clustering...")
        start = time.time()
        self._build_kmeans()
        print(f"  Completed in {time.time() - start:.2f}s")
        
        # Step 2: Assign vectors to clusters (with overlap)
        print(f"\n{'─'*70}")
        print("Step 2: Overlapping cluster assignment...")
        start = time.time()
        self._build_overlapping_assignments()
        print(f"  Completed in {time.time() - start:.2f}s")
        
        # Step 3: Build Vamana graph over centroids
        print(f"\n{'─'*70}")
        print("Step 3: Building centroid Vamana graph...")
        start = time.time()
        self._build_centroid_vamana()
        print(f"  Completed in {time.time() - start:.2f}s")
        
        # Step 4: Build Vamana graph inside each cluster
        print(f"\n{'─'*70}")
        print("Step 4: Building intra-cluster Vamana graphs...")
        start = time.time()
        self._build_cluster_vamana_graphs()
        print(f"  Completed in {time.time() - start:.2f}s")
        
        # Step 5: Build inter-cluster connections
        print(f"\n{'─'*70}")
        print("Step 5: Building inter-cluster boundary connections...")
        start = time.time()
        self._build_inter_cluster_connections()
        print(f"  Completed in {time.time() - start:.2f}s")
        
        # Print statistics
        self._print_index_stats()
    
    def _build_kmeans(self):
        """K-means clustering"""
        if not HAS_SKLEARN:
            raise RuntimeError("sklearn required")
        
        X = self.vectors if HAS_NUMPY else np.array(self.vectors, dtype=np.float32)
        
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10, max_iter=100)
        self.cluster_labels = kmeans.fit_predict(X)
        self.centroids = kmeans.cluster_centers_
        
        print(f"  Created {self.n_clusters} clusters")
    
    def _build_overlapping_assignments(self):
        """Assign vectors to multiple clusters based on distance threshold"""
        self.posting_lists = {k: [] for k in range(self.n_clusters)}
        self.vector_to_clusters = {}
        
        for vec_id in range(self.ntotal):
            vec = self.vectors[vec_id]
            
            distances = [(self._distance(vec, self.centroids[k]), k) 
                        for k in range(self.n_clusters)]
            distances.sort()
            
            min_dist = distances[0][0]
            threshold = min_dist * self.expansion_factor
            
            assigned = []
            for dist, k in distances:
                if dist <= threshold:
                    self.posting_lists[k].append(vec_id)
                    assigned.append(k)
                else:
                    break
            
            self.vector_to_clusters[vec_id] = assigned
        
        # Statistics
        cluster_sizes = [len(self.posting_lists[k]) for k in range(self.n_clusters)]
        clusters_per_vec = [len(v) for v in self.vector_to_clusters.values()]
        
        print(f"  Cluster sizes: min={min(cluster_sizes)}, max={max(cluster_sizes)}, avg={sum(cluster_sizes)/len(cluster_sizes):.1f}")
        print(f"  Clusters per vector: min={min(clusters_per_vec)}, max={max(clusters_per_vec)}, avg={sum(clusters_per_vec)/len(clusters_per_vec):.2f}")
    
    def _build_centroid_vamana(self):
        """Build Vamana graph over centroids"""
        n = self.n_clusters
        R = min(self.R_centroid, n - 1)
        
        self.centroid_graph = {i: [] for i in range(n)}
        self.centroid_entry = random.randint(0, n - 1)
        
        node_order = list(range(n))
        random.shuffle(node_order)
        
        for idx, node_id in enumerate(node_order):
            if idx == 0:
                continue
            
            candidates = self._search_centroid_graph(
                self.centroids[node_id],
                min(self.build_L_centroid, n),
                exclude={node_id},
            )
            neighbors = self._alpha_prune_centroids(node_id, candidates, R)
            self.centroid_graph[node_id] = neighbors
            
            for neighbor in neighbors:
                if node_id not in self.centroid_graph[neighbor]:
                    self.centroid_graph[neighbor].append(node_id)
                    if len(self.centroid_graph[neighbor]) > R * 2:
                        cands = [(self._distance(self.centroids[neighbor], self.centroids[n]), n) 
                                 for n in self.centroid_graph[neighbor]]
                        self.centroid_graph[neighbor] = self._alpha_prune_centroids(neighbor, cands, R * 2)
        
        degrees = [len(self.centroid_graph[i]) for i in range(n)]
        print(f"  Centroid graph: {n} nodes, degree min={min(degrees)}, max={max(degrees)}, avg={sum(degrees)/len(degrees):.1f}")
    
    def _search_centroid_graph(self, query, L: int, exclude: Set[int] = None) -> List[Tuple[float, int]]:
        """Search centroid Vamana"""
        if exclude is None:
            exclude = set()
        
        visited = set()
        candidates = []
        
        entry_dist = self._distance(query, self.centroids[self.centroid_entry])
        heappush(candidates, (entry_dist, self.centroid_entry))
        visited.add(self.centroid_entry)
        
        result = [(entry_dist, self.centroid_entry)] if self.centroid_entry not in exclude else []
        
        while candidates and len(visited) < L:
            _, node = heappop(candidates)
            
            for neighbor in self.centroid_graph.get(node, []):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                neighbor_dist = self._distance(query, self.centroids[neighbor])
                heappush(candidates, (neighbor_dist, neighbor))
                if neighbor not in exclude:
                    result.append((neighbor_dist, neighbor))
        
        return sorted(result)
    
    def _alpha_prune_centroids(self, node_id: int, candidates: List[Tuple[float, int]], R: int) -> List[int]:
        """α-pruning for centroid graph"""
        candidates = sorted(candidates)
        result = []
        
        for dist, cand_id in candidates:
            if cand_id == node_id:
                continue
            
            should_add = True
            for existing_id in result:
                dist_existing = self._distance(self.centroids[cand_id], self.centroids[existing_id])
                if dist >= self.alpha * dist_existing:
                    should_add = False
                    break
            
            if should_add:
                result.append(cand_id)
                if len(result) >= R:
                    break
        
        return result
    
    def _build_cluster_vamana_graphs(self):
        """Build Vamana graph inside each cluster"""
        total_nodes = 0
        total_edges = 0
        
        for cluster_id in range(self.n_clusters):
            node_ids = self.posting_lists[cluster_id]
            n = len(node_ids)
            
            if n == 0:
                continue
            
            # Create local-global mappings
            local_to_global = {i: node_ids[i] for i in range(n)}
            global_to_local = {node_ids[i]: i for i in range(n)}
            
            # Build Vamana graph for this cluster
            R = min(self.R_cluster, n - 1)
            adjacency = {i: [] for i in range(n)}
            entry_point = random.randint(0, n - 1)
            
            if n > 1:
                node_order = list(range(n))
                random.shuffle(node_order)
                
                for idx, local_id in enumerate(node_order):
                    if idx == 0:
                        continue
                    
                    global_id = local_to_global[local_id]
                    candidates = self._search_cluster_graph_build(
                        cluster_id, local_to_global, adjacency, 
                        self.vectors[global_id], min(self.build_L_cluster, n), 
                        entry_point, exclude={local_id}
                    )
                    
                    neighbors = self._alpha_prune_cluster(
                        local_id, candidates, R, local_to_global
                    )
                    
                    adjacency[local_id] = neighbors
                    
                    for neighbor in neighbors:
                        if local_id not in adjacency[neighbor]:
                            adjacency[neighbor].append(local_id)
                            if len(adjacency[neighbor]) > R * 2:
                                cands = [(self._distance_by_id(local_to_global[neighbor], local_to_global[nn]), nn)
                                         for nn in adjacency[neighbor]]
                                adjacency[neighbor] = self._alpha_prune_cluster(neighbor, cands, R * 2, local_to_global)
            
            # Create cluster graph object
            self.cluster_graphs[cluster_id] = ClusterGraph(
                cluster_id=cluster_id,
                node_ids=node_ids,
                local_to_global=local_to_global,
                global_to_local=global_to_local,
                adjacency=adjacency,
                entry_point=entry_point,
                boundary_nodes=set(),
                boundary_connections={}
            )
            
            total_nodes += n
            total_edges += sum(len(adj) for adj in adjacency.values())
        
        print(f"  Built {len(self.cluster_graphs)} cluster graphs")
        print(f"  Total nodes: {total_nodes}, Total edges: {total_edges}")
    
    def _search_cluster_graph_build(self, cluster_id: int, local_to_global: Dict, 
                                     adjacency: Dict, query, L: int, 
                                     entry: int, exclude: Set[int] = None) -> List[Tuple[float, int]]:
        """Search within cluster during build"""
        if exclude is None:
            exclude = set()
        
        visited = set()
        candidates = []
        
        entry_global = local_to_global[entry]
        entry_dist = self._distance(query, self.vectors[entry_global])
        heappush(candidates, (entry_dist, entry))
        visited.add(entry)
        
        result = [(entry_dist, entry)] if entry not in exclude else []
        
        while candidates and len(visited) < L:
            _, node = heappop(candidates)
            
            for neighbor in adjacency.get(node, []):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                neighbor_global = local_to_global[neighbor]
                neighbor_dist = self._distance(query, self.vectors[neighbor_global])
                heappush(candidates, (neighbor_dist, neighbor))
                if neighbor not in exclude:
                    result.append((neighbor_dist, neighbor))
        
        return sorted(result)
    
    def _alpha_prune_cluster(self, node_id: int, candidates: List[Tuple[float, int]], 
                             R: int, local_to_global: Dict) -> List[int]:
        """α-pruning for cluster graph"""
        candidates = sorted(candidates)
        result = []
        
        for dist, cand_id in candidates:
            if cand_id == node_id:
                continue
            
            should_add = True
            for existing_id in result:
                dist_existing = self._distance_by_id(
                    local_to_global[cand_id], 
                    local_to_global[existing_id]
                )
                if dist >= self.alpha * dist_existing:
                    should_add = False
                    break
            
            if should_add:
                result.append(cand_id)
                if len(result) >= R:
                    break
        
        return result
    
    def _build_inter_cluster_connections(self):
        """Build connections between overlapping clusters via boundary nodes"""
        total_boundary_nodes = 0
        total_inter_connections = 0
        
        for cluster_id, cg in self.cluster_graphs.items():
            n = len(cg.node_ids)
            if n == 0:
                continue
            
            # Identify boundary nodes (nodes that belong to multiple clusters)
            boundary_local_ids = []
            for local_id in range(n):
                global_id = cg.local_to_global[local_id]
                if len(self.vector_to_clusters[global_id]) > 1:
                    boundary_local_ids.append(local_id)
            
            # Limit boundary nodes to a fraction of cluster
            max_boundary = max(1, int(n * self.boundary_fraction))
            if len(boundary_local_ids) > max_boundary:
                # Keep nodes with most cluster memberships
                boundary_local_ids.sort(
                    key=lambda lid: len(self.vector_to_clusters[cg.local_to_global[lid]]),
                    reverse=True
                )
                boundary_local_ids = boundary_local_ids[:max_boundary]
            
            cg.boundary_nodes = set(boundary_local_ids)
            total_boundary_nodes += len(boundary_local_ids)
            
            # Build inter-cluster connections
            cg.boundary_connections = {}
            for local_id in boundary_local_ids:
                global_id = cg.local_to_global[local_id]
                other_clusters = [c for c in self.vector_to_clusters[global_id] if c != cluster_id]
                
                connections = []
                for other_cluster in other_clusters:
                    other_cg = self.cluster_graphs.get(other_cluster)
                    if other_cg and global_id in other_cg.global_to_local:
                        other_local_id = other_cg.global_to_local[global_id]
                        connections.append((other_cluster, other_local_id))
                        total_inter_connections += 1
                
                if connections:
                    cg.boundary_connections[local_id] = connections
        
        print(f"  Total boundary nodes: {total_boundary_nodes}")
        print(f"  Total inter-cluster connections: {total_inter_connections}")
    
    def _print_index_stats(self):
        """Print index statistics"""
        print(f"\n{'='*70}")
        print(f"{'DISKANN INDEX STATISTICS':^70}")
        print(f"{'='*70}")
        
        print(f"\nDataset: {self.ntotal:,} vectors, {self.dimension} dimensions")
        print(f"Clusters: {self.n_clusters}")
        
        # Cluster sizes
        cluster_sizes = [len(cg.node_ids) for cg in self.cluster_graphs.values()]
        print(f"\nCluster sizes: min={min(cluster_sizes)}, max={max(cluster_sizes)}, avg={sum(cluster_sizes)/len(cluster_sizes):.1f}")
        
        # Cluster graph stats
        total_cluster_edges = sum(
            sum(len(adj) for adj in cg.adjacency.values()) 
            for cg in self.cluster_graphs.values()
        )
        print(f"Total intra-cluster edges: {total_cluster_edges:,}")
        
        # Boundary stats
        total_boundary = sum(len(cg.boundary_nodes) for cg in self.cluster_graphs.values())
        total_inter = sum(
            sum(len(conns) for conns in cg.boundary_connections.values())
            for cg in self.cluster_graphs.values()
        )
        print(f"Boundary nodes: {total_boundary:,}")
        print(f"Inter-cluster connections: {total_inter:,}")
        
        print(f"{'='*70}\n")
    
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
    
    def search(self, query, k: int = 10, nprobe: int = 4, 
               L_centroid: int = 50, L_cluster: int = 50,
               allow_inter_cluster: bool = True,
               L_total: Optional[int] = None) -> Tuple[List[Tuple[int, float]], dict]:
        """
        DiskANN search with full hop tracking.
        
        Args:
            query: Query vector
            k: Number of results
            nprobe: Number of clusters to search
            L_centroid: Beam width for centroid search
            L_cluster: Beam width for intra-cluster search
            allow_inter_cluster: Whether to follow inter-cluster edges
            L_total: Global visit budget (overrides L_cluster * nprobe)
        """
        result_stats = {
            'centroid_hops': 0,
            'clusters_accessed': 0,
            'intra_cluster_hops': 0,
            'inter_cluster_hops': 0,
            'vectors_visited': 0,
            'total_hops': 0,
        }
        
        # Phase 1: Find relevant clusters via centroid Vamana
        visited_centroids = set()
        centroid_candidates = []
        
        entry_dist = self._distance(query, self.centroids[self.centroid_entry])
        heappush(centroid_candidates, (entry_dist, self.centroid_entry))
        visited_centroids.add(self.centroid_entry)
        result_stats['centroid_hops'] += 1
        
        centroid_results = [(entry_dist, self.centroid_entry)]
        
        while centroid_candidates and len(visited_centroids) < L_centroid:
            _, node = heappop(centroid_candidates)
            
            for neighbor in self.centroid_graph.get(node, []):
                if neighbor in visited_centroids:
                    continue
                visited_centroids.add(neighbor)
                result_stats['centroid_hops'] += 1
                
                neighbor_dist = self._distance(query, self.centroids[neighbor])
                heappush(centroid_candidates, (neighbor_dist, neighbor))
                centroid_results.append((neighbor_dist, neighbor))
        
        centroid_results.sort()
        top_clusters = [cid for _, cid in centroid_results[:nprobe]]
        result_stats['clusters_accessed'] = len(top_clusters)
        
        # Phase 2: Search within clusters with inter-cluster traversal
        global_visited = set()
        global_candidates = []  # (distance, global_id)
        clusters_actually_visited = set(top_clusters)
        
        # Initialize from all top clusters
        for cluster_id in top_clusters:
            cg = self.cluster_graphs.get(cluster_id)
            if cg is None or len(cg.node_ids) == 0:
                continue
            
            entry_global = cg.local_to_global[cg.entry_point]
            if entry_global not in global_visited:
                global_visited.add(entry_global)
                result_stats['intra_cluster_hops'] += 1
                entry_dist = self._distance(query, self.vectors[entry_global])
                heappush(global_candidates, (entry_dist, entry_global, cluster_id, cg.entry_point))
        
        # Unified beam search across clusters
        final_candidates = []
        
        visit_budget = (L_total if L_total is not None else (L_cluster * nprobe))
        while global_candidates and len(global_visited) < visit_budget:
            dist, global_id, current_cluster, local_id = heappop(global_candidates)
            
            # Add to results
            if len(final_candidates) < k:
                heappush(final_candidates, (-dist, global_id))
            elif dist < -final_candidates[0][0]:
                heapreplace(final_candidates, (-dist, global_id))
            
            cg = self.cluster_graphs[current_cluster]
            
            # Explore intra-cluster neighbors
            for neighbor_local in cg.adjacency.get(local_id, []):
                neighbor_global = cg.local_to_global[neighbor_local]
                if neighbor_global in global_visited:
                    continue
                
                global_visited.add(neighbor_global)
                result_stats['intra_cluster_hops'] += 1
                
                neighbor_dist = self._distance(query, self.vectors[neighbor_global])
                heappush(global_candidates, (neighbor_dist, neighbor_global, current_cluster, neighbor_local))
            
            # Explore inter-cluster connections if enabled
            if allow_inter_cluster and local_id in cg.boundary_connections:
                for other_cluster, other_local in cg.boundary_connections[local_id]:
                    if other_cluster not in clusters_actually_visited:
                        clusters_actually_visited.add(other_cluster)
                    
                    other_cg = self.cluster_graphs.get(other_cluster)
                    if other_cg is None:
                        continue
                    
                    # The same global_id, just in different cluster context
                    # Explore neighbors from that cluster
                    for neighbor_local in other_cg.adjacency.get(other_local, []):
                        neighbor_global = other_cg.local_to_global[neighbor_local]
                        if neighbor_global in global_visited:
                            continue
                        
                        global_visited.add(neighbor_global)
                        result_stats['inter_cluster_hops'] += 1
                        
                        neighbor_dist = self._distance(query, self.vectors[neighbor_global])
                        heappush(global_candidates, (neighbor_dist, neighbor_global, other_cluster, neighbor_local))
        
        result_stats['vectors_visited'] = len(global_visited)
        result_stats['total_hops'] = (result_stats['centroid_hops'] + 
                                       result_stats['intra_cluster_hops'] + 
                                       result_stats['inter_cluster_hops'])
        result_stats['clusters_accessed'] = len(clusters_actually_visited)
        
        # Extract final results
        results = []
        while final_candidates:
            neg_dist, global_id = heappop(final_candidates)
            results.append((global_id, -neg_dist))
        results.sort(key=lambda x: x[1])
        
        return results[:k], result_stats
    
    def run_queries(self, queries_path: str, num_queries: int = 100,
                    k: int = 10, nprobe: int = 4, 
                    L_centroid: int = 50, L_cluster: int = 50,
                    allow_inter_cluster: bool = True, gt_path: str = None,
                    L_total: Optional[int] = None):
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
        print(f"  Running {num_queries} queries")
        if L_total is None:
            print(f"  k={k}, nprobe={nprobe}, L_centroid={L_centroid}, L_cluster={L_cluster}")
        else:
            print(f"  k={k}, nprobe={nprobe}, L_centroid={L_centroid}, L_cluster={L_cluster}, L_total={L_total}")
        print(f"  Inter-cluster traversal: {'Enabled' if allow_inter_cluster else 'Disabled'}")
        
        # Reset statistics
        self.stats = DiskANNStats()
        all_results = []
        
        start_time = time.time()
        for i in range(num_queries):
            if (i + 1) % 100 == 0 or i == num_queries - 1:
                print(f"\r  Progress: {i + 1}/{num_queries}", end="", flush=True)
            
            query = queries[i]
            results, query_stats = self.search(
                query, k=k, nprobe=nprobe, 
                L_centroid=L_centroid, L_cluster=L_cluster,
                allow_inter_cluster=allow_inter_cluster,
                L_total=L_total,
            )
            
            all_results.append(results)
            self.stats.centroid_search_hops.append(query_stats['centroid_hops'])
            self.stats.clusters_accessed_per_query.append(query_stats['clusters_accessed'])
            self.stats.intra_cluster_hops.append(query_stats['intra_cluster_hops'])
            self.stats.inter_cluster_hops.append(query_stats['inter_cluster_hops'])
            self.stats.vectors_visited_per_query.append(query_stats['vectors_visited'])
            self.stats.total_hops_per_query.append(query_stats['total_hops'])
        
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
        """Print comprehensive statistics"""
        print(f"\n{'='*100}")
        print(f"DISKANN HOP STATISTICS: {dataset_name}" if dataset_name else "DISKANN HOP STATISTICS")
        print(f"{'='*100}")
        
        num_queries = len(self.stats.total_hops_per_query)
        
        # ==================== CLUSTER ACCESS (KEY METRIC!) ====================
        print(f"\n{'*'*100}")
        print(f"{'*** CLUSTER ACCESS STATISTICS ***':^100}")
        print(f"{'*'*100}")
        
        clusters = self.stats.clusters_accessed_per_query
        print(f"  Average clusters accessed: {sum(clusters)/len(clusters):.2f}")
        print(f"  Minimum clusters accessed: {min(clusters)}")
        print(f"  Maximum clusters accessed: {max(clusters)}")
        
        from collections import Counter
        cluster_counts = Counter(clusters)
        print(f"\n  Distribution:")
        for count in sorted(cluster_counts.keys()):
            pct = cluster_counts[count] / len(clusters) * 100
            bar = '█' * int(pct / 2)
            print(f"    {count:3d} clusters: {cluster_counts[count]:4d} queries ({pct:5.1f}%) {bar}")
        
        # ==================== HOP BREAKDOWN ====================
        print(f"\n{'-'*100}")
        print(f"{'HOP BREAKDOWN BY COMPONENT':^100}")
        print(f"{'-'*100}")
        
        # Table header
        print(f"{'Component':<40} {'Avg':<12} {'Min':<10} {'Max':<10} {'Total':<15}")
        print(f"{'-'*100}")
        
        # Centroid hops
        centroid = self.stats.centroid_search_hops
        print(f"{'Centroid Vamana search':<40} {sum(centroid)/len(centroid):<12.2f} {min(centroid):<10} {max(centroid):<10} {sum(centroid):<15,}")
        
        # Intra-cluster hops
        intra = self.stats.intra_cluster_hops
        print(f"{'Intra-cluster Vamana search':<40} {sum(intra)/len(intra):<12.2f} {min(intra):<10} {max(intra):<10} {sum(intra):<15,}")
        
        # Inter-cluster hops
        inter = self.stats.inter_cluster_hops
        print(f"{'Inter-cluster traversal':<40} {sum(inter)/len(inter):<12.2f} {min(inter):<10} {max(inter):<10} {sum(inter):<15,}")
        
        # Total
        total = self.stats.total_hops_per_query
        print(f"{'-'*100}")
        print(f"{'TOTAL HOPS':<40} {sum(total)/len(total):<12.2f} {min(total):<10} {max(total):<10} {sum(total):<15,}")
        
        # ==================== VECTORS VISITED ====================
        print(f"\n{'-'*100}")
        print(f"VECTORS VISITED")
        print(f"{'-'*100}")
        
        visited = self.stats.vectors_visited_per_query
        print(f"  Average vectors visited: {sum(visited)/len(visited):.2f}")
        print(f"  Min: {min(visited)}, Max: {max(visited)}")
        print(f"  Selectivity: {sum(visited)/len(visited)/self.ntotal*100:.4f}% of dataset per query")
        
        # ==================== EFFICIENCY ANALYSIS ====================
        print(f"\n{'-'*100}")
        print(f"EFFICIENCY ANALYSIS")
        print(f"{'-'*100}")
        
        avg_clusters = sum(clusters) / len(clusters)
        avg_intra = sum(intra) / len(intra)
        avg_inter = sum(inter) / len(inter)
        avg_total = sum(total) / len(total)
        
        print(f"  Hops per cluster accessed: {(avg_intra + avg_inter) / avg_clusters:.2f}")
        print(f"  Inter-cluster fraction: {avg_inter / (avg_intra + avg_inter) * 100:.1f}%")
        
        # ==================== RECALL & SPEED STATISTICS ====================
        print(f"\n{'-'*100}")
        print(f"RECALL & SPEED STATISTICS")
        print(f"{'-'*100}")
        
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
        
        print(f"{'='*100}\n")
        
        result = {
            'dataset_size': self.ntotal,
            'n_clusters': self.n_clusters,
            'num_queries': num_queries,
            'avg_clusters_accessed': sum(clusters) / len(clusters),
            'min_clusters_accessed': min(clusters),
            'max_clusters_accessed': max(clusters),
            'avg_centroid_hops': sum(centroid) / len(centroid),
            'avg_intra_cluster_hops': sum(intra) / len(intra),
            'avg_inter_cluster_hops': sum(inter) / len(inter),
            'avg_total_hops': sum(total) / len(total),
            'avg_vectors_visited': sum(visited) / len(visited),
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
    parser = argparse.ArgumentParser(description='DiskANN-Style Partitioned Search')
    parser.add_argument('--vectors', type=str, required=True, help='Base vectors')
    parser.add_argument('--queries', type=str, required=True, help='Query vectors')
    parser.add_argument('--gt', type=str, default=None, help='Ground truth .ivecs file')
    parser.add_argument('--num-queries', type=int, default=100)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--n-clusters', type=int, default=None)
    parser.add_argument('--nprobe', type=int, default=4)
    parser.add_argument('--L-centroid', type=int, default=50)
    parser.add_argument('--L-cluster', type=int, default=50)
    parser.add_argument('--L-total', type=int, default=None, help='Global visit budget (overrides L_cluster * nprobe)')
    parser.add_argument('--expansion', type=float, default=1.3)
    parser.add_argument('--R-centroid', type=int, default=32)
    parser.add_argument('--R-cluster', type=int, default=32)
    parser.add_argument('--alpha', type=float, default=1.2)
    parser.add_argument('--boundary-fraction', type=float, default=0.2)
    parser.add_argument('--build-L-centroid', type=int, default=200)
    parser.add_argument('--build-L-cluster', type=int, default=100)
    parser.add_argument('--no-inter-cluster', action='store_true')
    parser.add_argument('--metric', type=str, default='l2')
    parser.add_argument('--name', type=str, default='')
    
    args = parser.parse_args()
    
    index = DiskANNIndex(
        vectors_path=args.vectors,
        metric=args.metric,
        n_clusters=args.n_clusters,
        expansion_factor=args.expansion,
        R_centroid=args.R_centroid,
        R_cluster=args.R_cluster,
        alpha=args.alpha,
        boundary_fraction=args.boundary_fraction,
        build_L_centroid=args.build_L_centroid,
        build_L_cluster=args.build_L_cluster,
    )
    
    index.build_index()
    
    index.run_queries(
        queries_path=args.queries,
        num_queries=args.num_queries,
        k=args.k,
        nprobe=args.nprobe,
        L_centroid=args.L_centroid,
        L_cluster=args.L_cluster,
        allow_inter_cluster=not args.no_inter_cluster,
        gt_path=args.gt,
        L_total=args.L_total,
    )
    
    index.print_statistics(dataset_name=args.name)


def run_all_datasets():
    """Run all datasets"""
    datasets = [
        {'name': '1k', 'vectors': 'data/1k/laion_base.fvecs', 
         'queries': 'data/1k/laion_query.fvecs', 'gt': 'data/1k/gt.ivecs',
         'n_clusters': 32, 'nprobe': 4},
        {'name': '10k', 'vectors': 'data/10k/laion_base.fvecs',
         'queries': 'data/laion_query.fvecs', 'gt': 'data/10k/gt.ivecs',
         'n_clusters': 100, 'nprobe': 8},
        {'name': '100k', 'vectors': 'data/100k/laion_base.fvecs',
         'queries': 'data/laion_query.fvecs', 'gt': 'data/100k/gt.ivecs',
         'n_clusters': 316, 'nprobe': 8},
        {'name': '1M', 'vectors': 'data/1m/laion_base.fvecs',
         'queries': 'data/laion_query.fvecs', 'gt': 'data/1m/gt.ivecs',
         'n_clusters': 1000, 'nprobe': 16},
    ]
    
    all_results = {}
    
    for ds in datasets:
        print(f"\n{'#'*100}")
        print(f"# Dataset: {ds['name']}")
        print(f"{'#'*100}")
        
        try:
            index = DiskANNIndex(
                vectors_path=ds['vectors'],
                metric='l2',
                n_clusters=ds['n_clusters'],
                expansion_factor=1.3
            )
            index.build_index()
            index.run_queries(
                queries_path=ds['queries'],
                num_queries=100, k=10, nprobe=ds['nprobe'],
                gt_path=ds.get('gt')
            )
            all_results[ds['name']] = index.print_statistics(dataset_name=ds['name'])
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Comparison
    print(f"\n{'='*140}")
    print(f"{'DISKANN COMPARISON':^140}")
    print(f"{'='*140}")
    print(f"{'Dataset':<8} {'Size':>10} {'Clusters':>8} {'AvgClust':>8} {'Centroid':>10} {'Intra':>10} {'Inter':>10} {'Total':>10} {'Recall@10':>10} {'QPS':>10}")
    print(f"{'-'*140}")
    for name, r in all_results.items():
        recall = r.get('recall_avg', 0)
        qps = r.get('qps', 0)
        print(f"{name:<8} {r['dataset_size']:>10,} {r['n_clusters']:>8} {r['avg_clusters_accessed']:>8.2f} {r['avg_centroid_hops']:>10.2f} {r['avg_intra_cluster_hops']:>10.2f} {r['avg_inter_cluster_hops']:>10.2f} {r['avg_total_hops']:>10.2f} {recall:>10.4f} {qps:>10.2f}")
    print(f"{'='*140}\n")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        run_all_datasets()
    else:
        main()