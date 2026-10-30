import json
import time
import math
import random
import argparse
from typing import Dict, List, Tuple, Optional, Union
from heapq import heappush, heappop, heapreplace, nlargest
from itertools import count
from collections import deque, OrderedDict

import sys
import os
import pathlib
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
path_to_python_files = str(pathlib.Path().resolve().parent) + "/python-implementation"
sys.path.append(path_to_python_files)

# Import from your existing modules
from oblivious_graph import OGraph, Vertex, FanOut
from smart_pointer import SmartPointer
from single_access_machine import (
    get_global_counter,
    print_path_oram_stash_stats,
    reset_write_batches,
    set_path_oram_bucket_capacity,
    set_path_oram_rary_read_evictions,
    set_path_oram_tree_size,
)
from smart_queue import SmartQueue as queue
from smart_stack import SmartStack as stack
from smart_avl_tree import SmartAVLTree
from smart_pointer_multi_write import SmartPointerMultiWrite

# Optional FAISS and ML libraries
_FAISS_AVAILABLE = True
_SKLEARN_AVAILABLE = True
try:
    import faiss
    import numpy as np
except Exception:
    _FAISS_AVAILABLE = False
    np = None

try:
    from sklearn.cluster import KMeans
except Exception:
    _SKLEARN_AVAILABLE = False

# Optional METIS graph partitioner
_METIS_AVAILABLE = True
try:
    import pymetis
except Exception:
    _METIS_AVAILABLE = False
    pymetis = None

# Clustering methods that partition nodes into cluster OSAM objects
# (as opposed to Louvain, which uses the parent OGraph clustering machinery)
NODE_CLUSTERED_METHODS = ('kmeans', 'metis', 'singleton')

# Import hnsw functions directly
try:
    from hnsw import load_fvecs
except ImportError:
    def load_fvecs(path: str):
        """Simple fallback fvecs loader"""
        import struct
        
        with open(path, 'rb') as f:
            vectors = []
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
            return vectors, n, d


class LogicalSearchRoundTripCounter:
    """Counter for logical search round trips (similar to oblivious_graph.py pattern)"""
    def __init__(self):
        self.search_rounds = 0
        self.current_query_rounds = 0
        self.per_query_rounds = []
        self.in_search = False
    
    def start_query(self):
        """Start counting rounds for a new query."""
        self.current_query_rounds = 0
        self.in_search = True
    
    def end_query(self):
        """End counting for current query."""
        if self.in_search:
            self.per_query_rounds.append(self.current_query_rounds)
            self.search_rounds += self.current_query_rounds
            self.in_search = False
    
    def increment_search_round(self):
        """Count one logical search round trip."""
        if self.in_search:
            self.current_query_rounds += 1
    
    def get_stats(self):
        """Get search-only statistics."""
        if not self.per_query_rounds:
            return {
                'total_rounds': 0,
                'avg_per_query': 0,
                'max_per_query': 0,
                'min_per_query': 0,
                'num_queries': 0
            }
        
        return {
            'total_rounds': self.search_rounds,
            'avg_per_query': sum(self.per_query_rounds) / len(self.per_query_rounds),
            'max_per_query': max(self.per_query_rounds),
            'min_per_query': min(self.per_query_rounds),
            'num_queries': len(self.per_query_rounds)
        }
    
    def reset(self):
        """Reset all counters."""
        self.search_rounds = 0
        self.current_query_rounds = 0
        self.per_query_rounds = []
        self.in_search = False


# Global logical search counter
logical_search_counter = LogicalSearchRoundTripCounter()


# ==================== CLUSTER POINTEE ====================

class HNSWCluster:
    """
    OSAM pointee holding one base-layer partition (K-means / METIS).

    Plays the role Cluster plays in oblivious_graph.py: the whole partition is
    a single object behind one SmartPointer.

    Cross-cluster edges store (dst_id, SmartPointer to dst's cluster) rather
    than a bare dst_id. A bare id is not a location, so following such an edge
    forced the client to translate it first: node_clusters[dst] to learn the
    cluster, then the entry_points AVL tree to learn that cluster's pointer.
    Carrying the pointer on the edge removes both steps -- and with them the
    need to read the plaintext node_clusters map on every hop.
    """

    def __init__(self, cluster_id, nodes, neighbors=None, cross=None):
        """Initialize an HNSWCluster"""
        self.type = "HNSW_CLUSTER"
        self.cluster_id = cluster_id      # travels inside the pointee; the cache key
        self.nodes = nodes                # List[int]
        self.neighbors = neighbors if neighbors is not None else {}   # src_id -> [dst_id] (intra)
        self.cross = cross if cross is not None else {}               # src_id -> [(dst_id, SmartPointer)]

    def __repr__(self) -> str:
        """String representation of an HNSWCluster"""
        return (f"HNSWCluster(type={self.type}, cluster_id={self.cluster_id}, "
                f"nodes={len(self.nodes)}, cross_srcs={len(self.cross)})")

    def smart_delete(self) -> None:
        """Delete an HNSWCluster; recurses dict -> list -> tuple -> SmartPointer"""
        SmartPointer.delete(self.cross)

    @staticmethod
    def smart_copy(cluster: "HNSWCluster", temp_copy: bool = False) -> "HNSWCluster":
        """Copy an HNSWCluster, copying every cross-cluster pointer it owns"""
        cross = {
            src: [(dst, SmartPointer.copy(p, temp_copy)) for dst, p in edges]
            for src, edges in cluster.cross.items()
        }
        neighbors = {src: list(dsts) for src, dsts in cluster.neighbors.items()}
        return HNSWCluster(cluster.cluster_id, list(cluster.nodes), neighbors, cross)


class HNSWUpperCluster:
    """
    OSAM pointee holding one partition of the *upper* layers (levels >= 1).

    HNSW's levels are nested: a node at level 2 also exists at level 1 and at
    level 0. So the upper index partitions the union of level >= 1 nodes ONCE
    and keeps every level's adjacency for a member in the same payload. A
    descent from level L to level L-1 at the same node then reads nothing --
    the payload in hand already answers both levels. Partitioning per level
    instead would charge a fetch for each level transition.

    neighbors/cross are keyed by level first, and mirror HNSWCluster: an intra
    edge is a bare dst_id, a cross edge carries the SmartPointer to the upper
    partition holding dst, so following it needs no lookup.

    descent maps a member to its level-0 object -- its HNSWCluster under
    kmeans/metis partitioning, its Vertex under no partitioning. That pointer
    is what hands the base-layer phase its entry, so a query never has to
    translate a node id into a location.
    """

    def __init__(self, cluster_id, nodes, neighbors=None, cross=None, descent=None):
        """Initialize an HNSWUpperCluster"""
        self.type = "HNSW_UPPER_CLUSTER"
        self.cluster_id = cluster_id
        self.nodes = nodes                                            # List[int]
        self.neighbors = neighbors if neighbors is not None else {}   # level -> {src: [dst]}
        self.cross = cross if cross is not None else {}               # level -> {src: [(dst, SmartPointer)]}
        self.descent = descent if descent is not None else {}         # node -> SmartPointer

    def __repr__(self) -> str:
        """String representation of an HNSWUpperCluster"""
        return (f"HNSWUpperCluster(type={self.type}, cluster_id={self.cluster_id}, "
                f"nodes={len(self.nodes)}, levels={sorted(self.neighbors)}, "
                f"descent={len(self.descent)})")

    def smart_delete(self) -> None:
        """Delete an HNSWUpperCluster; both pointer-bearing fields must be freed"""
        SmartPointer.delete(self.cross)
        SmartPointer.delete(self.descent)

    @staticmethod
    def smart_copy(cluster: "HNSWUpperCluster", temp_copy: bool = False) -> "HNSWUpperCluster":
        """Copy an HNSWUpperCluster, copying every pointer it owns"""
        cross = {
            level: {
                src: [(dst, SmartPointer.copy(p, temp_copy)) for dst, p in edges]
                for src, edges in by_src.items()
            }
            for level, by_src in cluster.cross.items()
        }
        neighbors = {
            level: {src: list(dsts) for src, dsts in by_src.items()}
            for level, by_src in cluster.neighbors.items()
        }
        descent = {
            node: SmartPointer.copy(p, temp_copy)
            for node, p in cluster.descent.items()
        }
        return HNSWUpperCluster(
            cluster.cluster_id, list(cluster.nodes), neighbors, cross, descent
        )


# ==================== GROUND TRUTH AND RECALL FUNCTIONS ====================

def load_ivecs(filename):
    """
    Load ground truth from .ivecs file.
    
    File format:
        Each record: [d, id1, id2, ..., id_d]
        where d = number of neighbors
    
    Returns:
        numpy array of shape (num_queries, k)
    """
    import struct
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Ground truth file not found: {filename}")
    
    gt = []
    with open(filename, 'rb') as f:
        while True:
            # Read dimension (4 bytes = int32)
            d_bytes = f.read(4)
            if not d_bytes:
                break
            
            d = struct.unpack('i', d_bytes)[0]
            
            # Read d integers (neighbors)
            neighbors = struct.unpack('i' * d, f.read(4 * d))
            gt.append(list(neighbors))
    
    return np.array(gt) if np is not None else gt


def load_ground_truth(gt_path, num_queries=None):
    """
    Load ground truth nearest neighbors from file.
    
    Args:
        gt_path: Path to ground truth file (.ivecs)
        num_queries: Limit number of queries (optional)
    
    Returns:
        list of lists: gt[query_idx] = [neighbor_ids...]
    """
    print(f"Loading ground truth from {gt_path}...")
    
    if gt_path.endswith('.ivecs'):
        gt_array = load_ivecs(gt_path)
    else:
        raise ValueError(f"Unsupported ground truth format: {gt_path}")
    
    # Convert to list of lists for easier indexing
    if isinstance(gt_array, np.ndarray):
        gt = [list(row) for row in gt_array]
    else:
        gt = gt_array
    
    if num_queries is not None:
        gt = gt[:num_queries]
    
    print(f"✓ Loaded ground truth: {len(gt)} queries, up to {len(gt[0])} neighbors per query")
    
    return gt


def compute_recall_metrics(ground_truth_ids, evaluated_result, k=10, gamma=1):
    """
    Calculate recall metrics following FAISS C++ implementation pattern.
    
    Args:
        ground_truth_ids: List of ground truth neighbor IDs [id1, id2, id3, ...]
                         (pre-computed exact neighbors from .ivecs file)
        evaluated_result: List of (id, dist) tuples from approximate search (O-HNSW)
        k: Number of neighbors to evaluate
        gamma: Tolerance factor for position matching
               gamma=1: strict (R@1 must be at position 0, R@10 at positions 0-9)
               gamma>1: relaxed (R@1 can be at positions 0 to gamma-1)
    
    Returns:
        dict with R@1, R@10, R@100 metrics
    """
    # Extract evaluated IDs (preserve order)
    eval_ids = [x[0] for x in evaluated_result[:k]]
    
    # Create ground truth sets for different recall levels
    # Note: ground_truth_ids is already sorted by distance (closest first)
    gt_size = len(ground_truth_ids)
    
    # Truncate ground truth to k/gamma as in C++ code
    gt_k_gamma = ground_truth_ids[:max(1, k // gamma)]
    
    gt_set_1 = set(ground_truth_ids[:1])      # Top-1 ground truth
    gt_set_10 = set(gt_k_gamma[:10])          # Top-10 ground truth (or k/gamma)
    gt_set_100 = set(gt_k_gamma[:min(100, len(gt_k_gamma))])  # Top-100 ground truth
    
    # Count matches within position tolerance (following C++ logic)
    n_1 = 0
    n_10 = 0
    n_100 = 0
    
    for j, eval_id in enumerate(eval_ids[:k]):
        # R@100: Top-100 ground truth found within position 100*gamma
        if eval_id in gt_set_100:
            if j < 100 * gamma:
                n_100 += 1
        
        # R@10: Top-10 ground truth found within position 10*gamma
        if eval_id in gt_set_10:
            if j < 10 * gamma:
                n_10 += 1
        
        # R@1: Top-1 ground truth found within position 1*gamma
        if eval_id in gt_set_1:
            if j < 1 * gamma:
                n_1 += 1
    
    return {
        'recall@1': n_1,      # Raw count (1 or 0 for single query)
        'recall@10': n_10,    # Raw count (0-10 for single query)
        'recall@100': n_100   # Raw count (0-100 for single query)
    }


def compute_average_recall(all_recalls):
    """
    Compute average recall across all queries.
    
    Args:
        all_recalls: List of recall dicts from compute_recall_metrics()
    
    Returns:
        dict with averaged recall metrics
    """
    nq = len(all_recalls)
    if nq == 0:
        return {'R@1': 0.0, 'R@10': 0.0, 'R@100': 0.0, 'num_queries': 0}
    
    total_r1 = sum(r['recall@1'] for r in all_recalls)
    total_r10 = sum(r['recall@10'] for r in all_recalls)
    total_r100 = sum(r['recall@100'] for r in all_recalls)
    
    return {
        'R@1': total_r1 / float(nq),
        'R@10': (total_r10 / float(nq))/10.0,
        'R@100': (total_r100 / float(nq))/100.0,
        'num_queries': nq
    }


class ObliviousHNSW(OGraph):
    """
    Oblivious HNSW with K-means clustering following oblivious_graph.py pattern.
    """
    
    def __init__(self, branching_factor=2, clustering_method=None, use_osam=True, make_copies=False,
                 n_clusters=None):
        # Initialize parent (DON'T enable clustering in parent for kmeans/metis)
        if clustering_method == 'louvain':
            use_graph_clustering = True
        else:
            use_graph_clustering = False
        
        super().__init__(
            branching_factor=branching_factor, 
            avl_tree_osam=use_osam
        )

        # Toggle whether copies should be created during traversal
        self.make_copies = make_copies
        
        # HNSW specific attributes
        self.max_level = 0
        self.entry_point = None
        self.vectors = {}
        self.faiss_index = None
        self.level_graphs = {}
        self.oblivious_upper_layers = True
        
        # Clustering configuration
        self.clustering_method = clustering_method  # None, 'louvain', 'kmeans', 'metis'

        # Requested number of clusters/partitions (None -> auto, sqrt(ntotal))
        self.n_clusters = n_clusters

        # True for methods that partition nodes into per-cluster OSAM objects
        self.use_node_clustering = clustering_method in NODE_CLUSTERED_METHODS

        # Opt-in: balance the reference tree of each cluster's cross-edge
        # pointers at build time (see _copy_cluster_ref). Off by default.
        self.balance_cluster_refs = False

        # Take every cross-edge pointer of a cluster in one batched copy instead
        # of one incremental copy per edge. Only the multi-write schemes support
        # num_copies > 1, so this is gated on the pointer label at build time.
        # Subsumes balance_cluster_refs: the batch installs a balanced subtree.
        self.batch_cluster_refs = True

        # How many clusters the client may hold at once during one query.
        # ('unbounded', None) keeps every cluster touched until the query ends,
        # which is what the search always did. See _resolve_cluster_cache_capacity.
        self.cluster_cache_spec = ('unbounded', None)
        self._cluster_cache_capacity = None  # resolved lazily; 0 means unbounded

        # Node-clustering data structures (matching oblivious_graph.py pattern)
        if self.use_node_clustering:
            self.use_kmeans_clustering = (clustering_method == 'kmeans')
            # These mirror communities in oblivious_graph.py
            self.node_clusters = {}  # node_id -> cluster_id (dict for faster access)
            self.cluster_nodes = {}  # cluster_id -> set of node_ids
            self.cluster_graph = {}  # cluster_id -> neighbors (cross-cluster edges)

        print(f"Initialized HNSW with clustering method: {clustering_method or 'None'}")
    
    def set_oblivious_upper_layers(self, oblivious: bool):
        """Set whether to use oblivious access for upper layers."""
        self.oblivious_upper_layers = oblivious
        mode = "Oblivious" if oblivious else "Non-oblivious"
        print(f"Upper layer access mode: {mode}")

    def _total_clusters(self):
        """Number of partitions, or 0 before partitioning has run."""
        return len(self.cluster_nodes) if getattr(self, 'cluster_nodes', None) else 0

    def _resolve_cluster_cache_capacity(self):
        """
        Turn cluster_cache_spec into a concrete number of clusters.

        Returns 0 for unbounded. A percentage needs the total cluster count,
        which only exists once partitioning has populated cluster_nodes, so this
        resolves lazily and memoises the answer.

        Never returns 0 for a real cap: the cluster being read has to stay
        resident, so 1 is the floor and is also the true no-caching setting.
        """
        if self._cluster_cache_capacity is not None:
            return self._cluster_cache_capacity

        kind, value = self.cluster_cache_spec

        if kind == 'percent':
            # entry_points is a SmartAVLTree; cluster_nodes is the plaintext
            # cluster_id -> members map, so the count comes from there
            total = self._total_clusters()
            capacity = max(1, math.ceil(value / 100.0 * total)) if total else 0
        elif kind == 'fixed':
            capacity = max(1, value)
        else:
            capacity = 0

        self._cluster_cache_capacity = capacity
        return capacity

    def describe_cluster_cache(self):
        """Human-readable form of the cluster cache setting, for the config banner."""
        kind, value = self.cluster_cache_spec
        total = self._total_clusters()

        if kind == 'unbounded':
            return f"unbounded (up to all {total} partitions)" if total else "unbounded"

        capacity = self._resolve_cluster_cache_capacity()
        spec = f"{value}%" if kind == 'percent' else ('off' if capacity == 1 else str(value))
        return f"{spec} -> {capacity} of {total} partitions"
    
    # ==================== K-MEANS CLUSTERING FOLLOWING OBLIVIOUS_GRAPH.PY PATTERN ====================
    
    def _build_base_layer_graph(self):
        """Extract base layer (level 0) as a graph structure."""
        base_edges = []
        
        if not self.level_graphs or 0 not in self.level_graphs:
            print("No base layer graph found")
            return base_edges
        
        base_layer = self.level_graphs[0]
        
        # Build edge list from adjacency lists
        for src_id, neighbors in base_layer.items():
            for dst_id in neighbors:
                base_edges.append((src_id, dst_id))
        
        return base_edges
    
    def _default_n_clusters(self, ntotal: int):
        """Number of clusters/partitions to request (CLI override, else sqrt(ntotal))."""
        if self.n_clusters is not None:
            return max(2, min(int(self.n_clusters), ntotal))
        return max(2, int(math.sqrt(ntotal)))

    @staticmethod
    def _print_cluster_size_stats(clusters):
        """Print size distribution for a cluster_id -> set(node_ids) mapping."""
        cluster_sizes = [len(nodes) for nodes in clusters.values()]

        if cluster_sizes and len(cluster_sizes) > 1:
            import statistics
            avg_size = sum(cluster_sizes) / len(cluster_sizes)
            std_size = statistics.stdev(cluster_sizes)
            var_size = statistics.variance(cluster_sizes)

            print(f"\nCluster size statistics:")
            print(f"  Count:        {len(cluster_sizes)}")
            print(f"  Min:          {min(cluster_sizes)}")
            print(f"  Max:          {max(cluster_sizes)}")
            print(f"  Mean:         {avg_size:.2f}")
            print(f"  Median:       {statistics.median(cluster_sizes)}")
            print(f"  Std dev:      {std_size:.2f}")
            print(f"  Variance:     {var_size:.2f}")
            print(f"  Coeff of var: {(std_size/avg_size*100):.1f}%")

    def _generate_kmeans_clusters(self, ntotal: int):
        """
        Generate K-means clusters from vectors.

        Returns: dict mapping cluster_id -> set of node_ids
        """
        if not self.vectors:
            print("No vectors available for K-means clustering")
            return {0: set(range(ntotal))}  # Single cluster fallback

        # Determine optimal number of clusters
        n_clusters = self._default_n_clusters(ntotal)

        # Prepare vectors for clustering
        node_ids = list(range(ntotal))
        vectors_array = []
        valid_nodes = []
        
        for node_id in node_ids:
            if node_id in self.vectors:
                vectors_array.append(self.vectors[node_id])
                valid_nodes.append(node_id)
        
        if len(vectors_array) < 2:
            return {0: set(node_ids)}
        
        # Perform K-means clustering on vectors
        if np is not None:
            X = np.array(vectors_array, dtype=np.float32)
        else:
            X = vectors_array
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        # Build cluster mapping
        clusters = {}
        for i, node_id in enumerate(valid_nodes):
            cluster_id = int(cluster_labels[i])
            if cluster_id not in clusters:
                clusters[cluster_id] = set()
            clusters[cluster_id].add(node_id)
        
        # Assign remaining nodes to nearest cluster
        if len(valid_nodes) < len(node_ids):
            cluster_centers = kmeans.cluster_centers_
            for node_id in node_ids:
                if node_id not in valid_nodes:
                    if node_id in self.vectors:
                        # Find nearest cluster
                        distances = [self._distance(self.vectors[node_id], center) 
                                   for center in cluster_centers]
                        best_cluster = int(np.argmin(distances) if np is not None else 
                                         distances.index(min(distances)))
                        clusters[best_cluster].add(node_id)
                    else:
                        # Random assignment
                        random_cluster = random.choice(list(clusters.keys()))
                        clusters[random_cluster].add(node_id)
        
        print(f"Generated {len(clusters)} K-means clusters")
        for cluster_id, nodes in clusters.items():
            print(f"  Cluster {cluster_id}: {len(nodes)} nodes")
        
        # Calculate cluster size statistics
        self._print_cluster_size_stats(clusters)

        return clusters

    def _generate_metis_clusters(self, ntotal: int):
        """
        Generate balanced graph partitions of the base layer with METIS.

        Unlike K-means (which clusters vectors in embedding space), METIS
        partitions the level-0 HNSW graph itself: it balances partition sizes
        while minimizing the number of edges cut between partitions.

        Returns: dict mapping cluster_id -> set of node_ids
        """
        if not _METIS_AVAILABLE:
            raise RuntimeError(
                "pymetis is required for METIS partitioning. "
                "Install with: pip install pymetis (requires libmetis)."
            )

        base_layer = self.level_graphs.get(0, {})
        if not base_layer:
            print("No base layer graph available for METIS partitioning")
            return {0: set(range(ntotal))}  # Single partition fallback

        # Build symmetrized (undirected) adjacency; METIS requires it
        adjacency_sets = [set() for _ in range(ntotal)]
        for node_id, neighbors in base_layer.items():
            if node_id >= ntotal:
                continue
            for neighbor in neighbors:
                if neighbor != node_id and neighbor < ntotal:
                    adjacency_sets[node_id].add(neighbor)
                    adjacency_sets[neighbor].add(node_id)

        adjacency_list = [sorted(neighbors) for neighbors in adjacency_sets]
        undirected_edges = sum(len(adj) for adj in adjacency_list) // 2

        n_parts = self._default_n_clusters(ntotal)
        if ntotal < n_parts * 2:
            n_parts = max(2, ntotal // 10)
            print(f"  Adjusted to {n_parts} partitions (graph too small)")

        print(f"Running METIS on base layer: {ntotal} nodes, "
              f"{undirected_edges} undirected edges, n_partitions={n_parts}")

        start_time = time.time()
        try:
            edge_cuts, partition_vector = pymetis.part_graph(n_parts, adjacency=adjacency_list)
            print(f"  METIS edge cuts: {edge_cuts}")
        except Exception as e:
            print(f"  METIS failed ({e}); falling back to round-robin partitioning")
            partition_vector = [i % n_parts for i in range(ntotal)]

        elapsed = time.time() - start_time

        # Reindex partition ids to a contiguous 0..k-1 range (METIS may leave gaps)
        raw = {}
        for node_id, part_id in enumerate(partition_vector):
            raw.setdefault(int(part_id), set()).add(node_id)

        clusters = {}
        for new_id, nodes in enumerate(v for v in raw.values() if v):
            clusters[new_id] = nodes

        print(f"Generated {len(clusters)} METIS partitions in {elapsed:.2f}s")
        for cluster_id, nodes in clusters.items():
            print(f"  Partition {cluster_id}: {len(nodes)} nodes")

        self._print_cluster_size_stats(clusters)

        return clusters


    def _generate_singleton_clusters(self, ntotal: int):
        """
        Give every node its own partition -- the non-partitioned baseline,
        built by the partitioned procedure.

        No METIS, no K-means: the assignment is the identity, so this is O(n)
        and needs neither pymetis nor the vectors.

        The point is what falls out downstream. With |P| = 1 every base-layer
        edge is a cross-partition edge, so each node's payload is one OSAM
        object holding its neighbour ids together with a SmartPointer to each
        neighbour's payload, and a hop is one dereference of that pointer.
        That is the access pattern random_walk/pagerank measure, and the one
        the old non-partitioned path never performs: it collected neighbour ids
        with add_neighbors(ds='SET'), which drops ogE and re-enters the next
        node by id through the entry_points AVL tree, so every dereference it
        made was on a single-alias FanOut pointer -- identical cost under
        r-ary and plain multi-write, which is why BOSAM and OSAM+ measured the
        same there.

        Returns: dict mapping cluster_id -> set(node_ids), cluster_id == node_id.
        """
        print(f"Assigning {ntotal} singleton partitions (one node each; "
              f"no METIS/K-means)")

        # No per-partition print and no size histogram: both are one line per
        # partition, and there are ntotal of them.
        return {node_id: {node_id} for node_id in range(ntotal)}


    def _build_cluster_graph_structure(self, base_edges, clusters):
        """
        Build inter-cluster graph structure (following oblivious_graph.py pattern).
        
        Returns:
            - intra_cluster_edges: dict mapping cluster_id -> list of edges within cluster
            - cross_cluster_edges: list of edges between clusters
        """
        # Map nodes to clusters
        node_to_cluster = {}
        for cluster_id, nodes in clusters.items():
            for node_id in nodes:
                node_to_cluster[node_id] = cluster_id
        
        # Separate intra-cluster and cross-cluster edges
        intra_cluster_edges = {cid: [] for cid in clusters.keys()}
        cross_cluster_edges = []
        
        for src, dst in base_edges:
            src_cluster = node_to_cluster.get(src)
            dst_cluster = node_to_cluster.get(dst)
            
            if src_cluster is None or dst_cluster is None:
                continue
            
            if src_cluster == dst_cluster:
                # Intra-cluster edge
                intra_cluster_edges[src_cluster].append((src, dst))
            else:
                # Cross-cluster edge
                cross_cluster_edges.append((src, dst, src_cluster, dst_cluster))
        
        return intra_cluster_edges, cross_cluster_edges, node_to_cluster
    
    def _copy_cluster_ref(self, cluster_id, root_ptr, refs_by_cluster):
        """
        Take one SmartPointer copy referencing cluster_id's payload, keeping
        that cluster's reference tree balanced. Opt-in via balance_cluster_refs.

        References to a pointee form a tree of addresses and deref walks a
        handle up to the root, so a handle's depth is what dereferencing it
        costs. Copying pushes the handle it copied from further down
        (immediately for multiwrite, once per full sibling group for the r-ary
        variant), so taking every copy from the same handle builds a chain as
        long as the number of cross edges entering the cluster -- up to ~2.8k
        on the 1k dataset.

        Round-robin builds the same number of references with the same allocs
        and writes, but at depth ~log(C) instead.
        """
        refs = refs_by_cluster.setdefault(cluster_id, deque([root_ptr]))
        src_ptr = refs.popleft()
        new_ptr = SmartPointer.copy(src_ptr)
        refs.append(src_ptr)
        refs.append(new_ptr)
        return new_ptr

    def _emulate_graph_clustered_partitioned(self, base_edges, ntotal):
        """
        Build oblivious clustered HNSW following oblivious_graph.py pattern.

        This matches the __emulate_graph_clustered() method in oblivious_graph.py:
        1. Generate clusters/partitions (replaces Louvain)
        2. Build intra-cluster structures in plaintext
        3. Store each cluster as single OSAM object
        4. Give every cross-cluster edge a SmartPointer to the destination cluster

        Cross edges form cycles between clusters, so no ordering lets us embed
        the pointers while first assembling the payloads. Hence allocate every
        cluster pointer first, then copy one per cross edge, then write the
        cross maps back into the already-stored payloads.

        Only step 1 depends on the partitioning method; everything downstream
        (OSAM packing, neighbor retrieval, search) is method-agnostic.
        """
        print(f"Building {self.clustering_method} clustered oblivious structure...")

        # Step 1: Generate clusters with the configured partitioning method
        if self.clustering_method == 'singleton':
            clusters = self._generate_singleton_clusters(ntotal)
        elif self.clustering_method == 'metis':
            clusters = self._generate_metis_clusters(ntotal)
        else:
            clusters = self._generate_kmeans_clusters(ntotal)

        # Step 2: Build cluster graph structure
        intra_cluster_edges, cross_cluster_edges, node_to_cluster = \
            self._build_cluster_graph_structure(base_edges, clusters)
        
        # Step 3: Store each cluster as a single OSAM object, cross map still empty
        # (matching oblivious_graph.py). Every cluster pointer must exist before
        # any cross edge can reference one.
        cluster_ptrs = {}

        for cluster_id, cluster_nodes in clusters.items():
            # Build intra-cluster neighbor structure
            cluster_neighbors = {}
            for src, dst in intra_cluster_edges[cluster_id]:
                if src not in cluster_neighbors:
                    cluster_neighbors[src] = []
                cluster_neighbors[src].append(dst)

            payload = HNSWCluster(
                cluster_id=cluster_id,
                nodes=list(cluster_nodes),
                neighbors=cluster_neighbors,
                cross={}  # Will fill this next
            )

            # Create OSAM pointer for entire cluster
            cluster_ptr = SmartPointer(SmartPointer.new(payload))
            self.entry_points[cluster_id] = cluster_ptr
            cluster_ptrs[cluster_id] = cluster_ptr

            # One live reference per cluster so far (its entry point)
            self.cluster_counts[cluster_id] = 1

            # Map each node to its cluster (like communities in oblivious_graph.py)
            for node_id in payload.nodes:
                self.node_clusters[node_id] = cluster_id
                self.v_ids[node_id] = node_id  # Store node IDs

        # Step 4: Replace each cross-cluster edge with a SmartPointer to the
        # destination cluster (one copy per edge, as oblivious_graph.py does for
        # every FanOut leaf that crosses a community boundary)
        cross_maps = {cluster_id: {} for cluster_id in clusters}
        refs_by_cluster = {}  # only used when balancing is enabled

        # Only the multi-write schemes service many aliases in one tree update;
        # every other pointer type raises NotImplementedError for num_copies > 1.
        # Note the class-level selector is "multiwriterary", while the instance
        # label on the same pointer reads "multiwrite_rary".
        use_batch = (
            self.batch_cluster_refs
            and not self.balance_cluster_refs
            and SmartPointer.label in ("multiwrite", "multiwriterary")
        )
        if self.balance_cluster_refs:
            cross_ref_mode = "round-robin"
        elif use_batch:
            cross_ref_mode = "batched"
        else:
            cross_ref_mode = "incremental"

        step4_a0, step4_r0, step4_w0 = get_global_counter()
        step4_t0 = time.perf_counter()

        # One batched copy per destination cluster, taken before any edge is
        # served (as oblivious_graph.py does for its FanOut leaves)
        batched_edge_ptrs: Dict[int, List[SmartPointer]] = {}
        if use_batch:
            copies_needed: Dict[int, int] = {}
            for _, _, _, dst_cluster in cross_cluster_edges:
                copies_needed[dst_cluster] = copies_needed.get(dst_cluster, 0) + 1

            for dst_cluster, num_copies in copies_needed.items():
                copies = SmartPointer.copy(
                    cluster_ptrs[dst_cluster], num_copies=num_copies
                )
                # num_copies == 1 returns a bare pointer, not a list
                batched_edge_ptrs[dst_cluster] = (
                    [copies] if num_copies == 1 else list(copies)
                )

        for src, dst, src_cluster, dst_cluster in cross_cluster_edges:
            if self.balance_cluster_refs:
                edge_ptr = self._copy_cluster_ref(
                    dst_cluster, cluster_ptrs[dst_cluster], refs_by_cluster
                )
            elif use_batch:
                edge_ptr = batched_edge_ptrs[dst_cluster].pop()
            else:
                edge_ptr = SmartPointer.copy(cluster_ptrs[dst_cluster])
            self.cluster_counts[dst_cluster] += 1
            cross_maps[src_cluster].setdefault(src, []).append((dst, edge_ptr))

        # Every batched copy must have been handed to exactly one edge
        assert all(not ptrs for ptrs in batched_edge_ptrs.values())

        step4_secs = time.perf_counter() - step4_t0
        step4_a1, step4_r1, step4_w1 = get_global_counter()
        step4_stats = (
            cross_ref_mode,
            step4_a1 - step4_a0,
            step4_r1 - step4_r0,
            step4_w1 - step4_w0,
            step4_secs,
        )

        # Step 5: Write the cross maps back into the stored payloads.
        # put_attr, never put: put deep-copies the value, which would duplicate
        # every cross pointer, and its default delete_old=True would smart_delete
        # the payload we are updating.
        for cluster_id, cross in cross_maps.items():
            SmartPointer.put_attr(cluster_ptrs[cluster_id], {"cross": cross}, delete_old=False)

        # Step 6: Store cluster metadata
        self.cluster_nodes = clusters
        
        print(f"Created {len(clusters)} cluster OSAM objects")
        print(f"Total nodes: {ntotal}")
        print(f"Intra-cluster edges: {sum(len(edges) for edges in intra_cluster_edges.values())}")
        print(f"Cross-cluster edges: {len(cross_cluster_edges)}")

        mode, d_alloc, d_read, d_write, secs = step4_stats
        print(
            f"Cross-edge pointer copies ({mode}): "
            f"{len(cross_cluster_edges)} edges, "
            f"allocs={d_alloc:,} reads={d_read:,} writes={d_write:,}, "
            f"{secs:.2f}s"
        )
        
        # max of the count of cross-cluster edges and max of the count of intra-cluster edges
        # Calculate max edges per cluster
        # max_intra_edges = max((len(edges) for edges in intra_cluster_edges.values()), default=0)
        
        # # Calculate cross-cluster edges per cluster
        # cross_cluster_counts = {}
        # for src, dst, src_cluster, dst_cluster in cross_cluster_edges:
        #     if src_cluster not in cross_cluster_counts:
        #         cross_cluster_counts[src_cluster] = 0
        #     cross_cluster_counts[src_cluster] += 1
        
        # max_cross_edges = max(cross_cluster_counts.values(), default=0) if cross_cluster_counts else 0
        
        # print(f"Max intra-cluster edges (one cluster): {max_intra_edges}")
        # print(f"Max cross-cluster edges (one cluster): {max_cross_edges}")
                # Calculate degree statistics per cluster
        max_degree_overall = 0
        avg_degree_per_cluster = {}
        
        for cluster_id, edges in intra_cluster_edges.items():
            # Count degree for each node in this cluster
            node_degrees = {}
            for src, dst in edges:
                node_degrees[src] = node_degrees.get(src, 0) + 1
            
            if node_degrees:
                max_degree = max(node_degrees.values())
                avg_degree = sum(node_degrees.values()) / len(node_degrees)
                
                max_degree_overall = max(max_degree_overall, max_degree)
                avg_degree_per_cluster[cluster_id] = {
                    'max': max_degree,
                    'avg': avg_degree,
                    'nodes': len(clusters[cluster_id])
                }
        
        # Calculate cross-cluster statistics
        cross_cluster_counts = {}
        for src, dst, src_cluster, dst_cluster in cross_cluster_edges:
            cross_cluster_counts[src_cluster] = cross_cluster_counts.get(src_cluster, 0) + 1
        
        max_cross_edges = max(cross_cluster_counts.values(), default=0) if cross_cluster_counts else 0
        avg_cross_edges = sum(cross_cluster_counts.values()) / len(cross_cluster_counts) if cross_cluster_counts else 0
        
        # Print summary
        print(f"\nDegree statistics:")
        print(f"  Max intra-cluster degree (per node): {max_degree_overall}")
        # Singleton partitions have no intra-cluster edges at all, so this dict
        # is empty and the mean would divide by zero.
        if avg_degree_per_cluster:
            avg_intra = (sum(c['avg'] for c in avg_degree_per_cluster.values())
                         / len(avg_degree_per_cluster))
            print(f"  Avg intra-cluster degree (per node): {avg_intra:.1f}")
        else:
            print(f"  Avg intra-cluster degree (per node): n/a (no intra-cluster edges)")
        print(f"\nCross-cluster edge statistics:")
        print(f"  Max cross-cluster edges (per cluster): {max_cross_edges}")
        print(f"  Avg cross-cluster edges (per cluster): {avg_cross_edges:.1f}")

    
    # ==================== OVERRIDE get_pointer FOR K-MEANS CLUSTERING ====================
    
    def get_pointer(self, src_name):
        """
        Get pointer for node following oblivious_graph.py pattern.
        
        For K-means/METIS: Returns pointer to the cluster containing the node.
        """
        if self.use_node_clustering:
            src_id = src_name if isinstance(src_name, int) else self.v_ids[src_name]
            cluster_id = self.node_clusters[src_id]  # Get cluster (like community)
            p = self.entry_points[cluster_id]  # Get cluster OSAM pointer
        elif self.use_clustering:  # Graph clustering (Louvain)
            src_id = self.v_ids[src_name]
            community = self.communities[src_id]
            p = self.entry_points[community]
        else:
            src_id = self.v_ids[src_name]
            p = self.entry_points[src_id]
        
        if p is None:
            raise KeyError(f"Error: Source {src_name} not found.")
        
        return src_id, p
    
    # ==================== CLUSTER-AWARE NEIGHBOR RETRIEVAL (K-MEANS / METIS) ====================

    def get_neighbors_in_cluster(self, node_id, cluster_data=None, with_pointers=False):
        """
        Get neighbors for node using the cluster structure.

        Method-agnostic: works for any partitioning that populates
        node_clusters + per-cluster OSAM objects (K-means, METIS).

        Following oblivious_graph.py pattern:
        1. Single OSAM access to get cluster data
        2. Extract intra-cluster neighbors (plaintext access)
        3. Extract cross-cluster connections (with cluster pointers)

        with_pointers=False returns a flat list of neighbor ids.
        with_pointers=True returns (intra_ids, [(dst_id, dst_cluster_pointer)]),
        so the search can follow a cross edge without looking the destination
        cluster up for itself.
        """
        # Get cluster ID for this node
        if node_id not in self.node_clusters:
            raise KeyError(f"Node {node_id} not in any cluster")

        # Single OSAM access to get cluster data (THIS IS THE KEY OPTIMIZATION)
        #!logical_search_counter.increment_search_round()
        if cluster_data is None:
            cluster_id = self.node_clusters[node_id]
            cluster_ptr = self.entry_points[cluster_id]
            cluster_data = SmartPointer.get(cluster_ptr)

        # Intra-cluster neighbors (plaintext access within cluster)
        intra = cluster_data.neighbors.get(node_id, ())

        # Cross-cluster neighbors, each carrying a pointer to its own cluster
        cross = cluster_data.cross.get(node_id, ())

        if with_pointers:
            return list(intra), list(cross)

        return list(intra) + [dst for dst, _ptr in cross]
    
    # ==================== NEIGHBOR RETRIEVAL WITH PROPER OBLIVIOUS/NON-OBLIVIOUS CONTROL ====================
    
    def _get_level_neighbors(self, node_id, level, nodes_dict):
        """
        Get neighbors at specific level with proper oblivious/non-oblivious handling.
        
        Decision matrix:
        
        BASE LAYER (level 0):
        - If K-means/METIS clustering enabled: Use cluster-based retrieval (always oblivious)
        - If graph clustering enabled: Use Louvain-based retrieval (always oblivious)
        - If no clustering: Respect oblivious_upper_layers setting

        UPPER LAYERS (level > 0):
        - K-means/METIS clustering does NOT apply to upper layers
        - Graph clustering does NOT apply to upper layers
        - Always respect oblivious_upper_layers setting
        """

        # ==================== BASE LAYER (Level 0) ====================
        if level == 0:
            # Node clustering: K-means / METIS (oblivious by design)
            if self.use_node_clustering:
                try:
                    return self.get_neighbors_in_cluster(node_id)
                except Exception as e:
                    print(f"{self.clustering_method} retrieval failed for node {node_id}: {e}")
                    # Fallback to direct access
                    if level in self.level_graphs and node_id in self.level_graphs[level]:
                        return self.level_graphs[level][node_id]
                    return []
            
            # Graph clustering (oblivious by design)
            elif self.use_clustering:
                try:
                    logical_search_counter.increment_search_round()
                    # Load pointer to Cluster
                    _, p_cluster = self.get_pointer(node_id)
                    logical_search_counter.increment_search_round()
                    neighbors_set = set()

                    # From Cluster, load entry pointer to Vertex
                    p_cluster, community, cluster_entry_points = self.get_cluster_attr(p_cluster, make_copies=self.make_copies)
                    p_node = cluster_entry_points[node_id]

                    # Search for neighbors
                    logical_search_counter.increment_search_round()
                    self.add_neighbors_clustered(p_node, 'SET', neighbors_set, make_copies=self.make_copies)
                    assert neighbors_set == nodes_dict[node_id]
                    
                    # Cleanup possible additional copies (depends on if make_copies is toggled)
                    for v_id, p in cluster_entry_points.items():
                        if SmartPointer.is_temp_copy(p):
                            self.ogV_counts[v_id] -= 1
                        SmartPointer.delete_temp_copy(p)

                    return list(neighbors_set)
                except Exception as e:
                    print(f"Graph clustering retrieval failed for node {node_id}: {e}")
                    if level in self.level_graphs and node_id in self.level_graphs[level]:
                        return self.level_graphs[level][node_id]
                    return []
            
            # NO CLUSTERING - use oblivious_upper_layers setting
            else:
                if self.oblivious_upper_layers:
                    # Oblivious access
                    try:
                        logical_search_counter.increment_search_round()
                        _, p_node = self.get_pointer(node_id)
                        logical_search_counter.increment_search_round()
                        neighbors_set = set()
                        self.add_neighbors(p_node, 'SET', neighbors_set, make_copies=self.make_copies)
                        assert neighbors_set == nodes_dict[node_id]
                        return list(neighbors_set)
                    except Exception as e:
                        print(f"Oblivious access failed for node {node_id}: {e}")
                        if level in self.level_graphs and node_id in self.level_graphs[level]:
                            return self.level_graphs[level][node_id]
                        return []
                else:
                    # Non-oblivious (direct) access
                    if level in self.level_graphs and node_id in self.level_graphs[level]:
                        return self.level_graphs[level][node_id]
                    return []
        
        # ==================== UPPER LAYERS (Level > 0) ====================
        # Clustering does NOT apply to upper layers
        # Always use oblivious_upper_layers setting
        
        if self.oblivious_upper_layers:
            # Oblivious access for upper layers
            # BUT: If node clustering is enabled, upper layers aren't in the OSAM structure
            # So we need to check if oblivious structure exists for upper layers

            if self.use_node_clustering:
                # K-means/METIS only apply to base layer
                # Upper layers must use direct access (not stored in OSAM when clustered)
                if level in self.level_graphs and node_id in self.level_graphs[level]:
                    return self.level_graphs[level][node_id]
                return []
            else:
                # No node clustering, can use standard oblivious access
                try:
                    logical_search_counter.increment_search_round()
                    _, p_node = self.get_pointer(node_id)
                    logical_search_counter.increment_search_round()
                    neighbors_set = set()
                    self.add_neighbors(p_node, 'SET', neighbors_set, make_copies=self.make_copies)
                    assert neighbors_set == nodes_dict[node_id]
                    return list(neighbors_set)
                except Exception as e:
                    # Fallback to direct access
                    if level in self.level_graphs and node_id in self.level_graphs[level]:
                        return self.level_graphs[level][node_id]
                    return []
        else:
            # Non-oblivious (direct) access for upper layers
            if level in self.level_graphs and node_id in self.level_graphs[level]:
                return self.level_graphs[level][node_id]
            return []
    
    # ==================== HNSW SEARCH WITH K-MEANS CLUSTERING ====================
    
    def oblivious_hnsw_search(self, query_vector, nodes_dict, k=10, ef_search=64, metric="ip"):
        """
        HNSW search with optional K-means clustering.
        
        Search strategy:
        1. Upper layers (max_level -> 1): Greedy search
           - Can be oblivious or non-oblivious based on oblivious_upper_layers
           - Does NOT use K-means clustering
        
        2. Base layer (level 0): ef-search
           - If K-means enabled: Uses cluster-based neighbor retrieval
           - If no K-means: Uses oblivious_upper_layers setting
        """
        if self.entry_point is None:
            raise RuntimeError("HNSW not loaded or entry point not set")
        
        logical_search_counter.start_query()
        
        try:
            current_closest = self.entry_point
            
            # Phase 1: Greedy search from max_level down to level 1
            # (Upper layers - can be oblivious or non-oblivious, NO K-means clustering)
            for level in range(self.max_level, 0, -1):
                try:
                    current_closest = self._oblivious_greedy_search_layer(
                        current_closest, query_vector, nodes_dict, level, metric
                    )
                except Exception as e:
                    print(f"Error in greedy search at level {level}: {e}")
                    import traceback
                    traceback.print_exc()
                    logical_search_counter.end_query()
                    return []
            
            # Phase 2: ef-search at base layer (level 0)
            # (Base layer - uses K-means clustering if enabled)
            try:
                results = self._oblivious_ef_search_layer_clustered(
                    current_closest, query_vector, nodes_dict, ef_search, k, metric
                )
            except Exception as e:
                print(f"Error in ef-search at base layer: {e}")
                import traceback
                traceback.print_exc()
                logical_search_counter.end_query()
                return []
            
            logical_search_counter.end_query()
            return results[:k]
            
        except Exception as e:
            logical_search_counter.end_query()
            print(f"Error in oblivious HNSW search: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _oblivious_greedy_search_layer(self, entry_node, query_vector, nodes_dict, level, metric):
        """
        Greedy search at upper levels.
        
        Upper layers (level > 0):
        - Uses standard HNSW greedy search
        - Can be oblivious or non-oblivious based on oblivious_upper_layers setting
        - Does NOT use K-means clustering (clustering only applies to base layer)
        """
        if entry_node not in self.vectors:
            print(f"Warning: Entry node {entry_node} not in vectors")
            return entry_node
        
        current_node = entry_node
        current_dist = self._distance(self.vectors[current_node], query_vector, metric)
        
        visited = set([current_node])
        improved = True
        
        while improved:
            improved = False
            
            try:
                # Get neighbors using appropriate method (oblivious or direct)
                neighbors = self._get_level_neighbors(current_node, level, nodes_dict)
            except Exception as e:
                print(f"Error getting neighbors for node {current_node} at level {level}: {e}")
                break
            
            for neighbor_id in neighbors:
                if neighbor_id in visited or neighbor_id not in self.vectors:
                    continue
                
                visited.add(neighbor_id)
                
                try:
                    neighbor_dist = self._distance(self.vectors[neighbor_id], query_vector, metric)
                except Exception as e:
                    print(f"Error calculating distance for neighbor {neighbor_id}: {e}")
                    continue
                
                if neighbor_dist < current_dist:
                    current_node = neighbor_id
                    current_dist = neighbor_dist
                    improved = True
                    break
        
        return current_node
    
    def _oblivious_ef_search_layer_clustered(self, entry_node, query_vector, nodes_dict, ef, k, metric):
        """
        ef-search at BASE LAYER (level 0) with K-means clustering optimization.
        
        Key behavior:
        - If K-means clustering enabled: Use cluster-based neighbor retrieval
        - If no clustering: Use oblivious_upper_layers setting for access method
        
        Key optimization with K-means:
        - When visiting a node, get its entire cluster with single OSAM access
        - Then explore all nodes in that cluster locally (no additional OSAM accesses)

        A candidate carries the pointer to the cluster it lives in, taken from
        the cross-cluster edge that discovered it, so following that edge needs
        no lookup. Only the entry node has no such pointer.
        """
        if entry_node not in self.vectors:
            print(f"Warning: Entry node {entry_node} not in vectors")
            return []

        visited_nodes = set()
        visited_clusters = set()  # Track visited clusters (for K-means)
        # LRU of the clusters the client currently holds. cluster_cache decides
        # residency, node_cache is the member -> cluster index into it; the two
        # are evicted together.
        cluster_cache = OrderedDict()  # cluster_id -> cluster data
        node_cache = {}  # node -> cached cluster data, so a hop can skip the fetch
        # One pointer per cluster seen this query, kept even after eviction so an
        # evicted cluster can be re-fetched without an entry_points AVL lookup
        cluster_ptrs = {}  # cluster_id -> pointer that reached it
        cache_capacity = self._resolve_cluster_cache_capacity()  # 0 == unbounded
        candidates = []
        dynamic_candidates = []

        # Keeps SmartPointers out of heap comparisons: distances tie often, and
        # a SmartPointer defines __eq__ but no __lt__
        tiebreak = count()

        # Initialize with entry node
        try:
            entry_dist = self._distance(self.vectors[entry_node], query_vector, metric)
        except Exception as e:
            print(f"Error calculating entry distance: {e}")
            return []

        def admit(neighbor_id, cluster_ptr):
            """Queue a neighbor, remembering the pointer to the cluster holding it"""
            if neighbor_id in visited_nodes or neighbor_id not in self.vectors:
                return

            visited_nodes.add(neighbor_id)

            try:
                neighbor_dist = self._distance(self.vectors[neighbor_id], query_vector, metric)
            except Exception as e:
                print(f"Error calculating distance for neighbor {neighbor_id}: {e}")
                return

            if len(dynamic_candidates) < ef:
                heappush(dynamic_candidates, (-neighbor_dist, neighbor_id))
                heappush(candidates, (neighbor_dist, next(tiebreak), neighbor_id, cluster_ptr))
            else:
                worst_dist = -dynamic_candidates[0][0]
                if neighbor_dist < worst_dist:
                    heapreplace(dynamic_candidates, (-neighbor_dist, neighbor_id))
                    heappush(candidates, (neighbor_dist, next(tiebreak), neighbor_id, cluster_ptr))

        heappush(candidates, (entry_dist, next(tiebreak), entry_node, None))
        heappush(dynamic_candidates, (-entry_dist, entry_node))
        visited_nodes.add(entry_node)

        while candidates:
            current_dist, _, current_node, cluster_ptr = heappop(candidates)

            # Early termination
            if len(dynamic_candidates) >= ef:
                worst_dist = -dynamic_candidates[0][0]
                if current_dist > worst_dist:
                    break

            # Get neighbors based on clustering method
            try:
                # ==================== NODE CLUSTERING (K-MEANS / METIS) ====================
                if self.use_node_clustering:
                    # Check if node is in a cluster
                    if current_node not in self.node_clusters:
                        print(f"Warning: Node {current_node} not in any cluster, using direct access")
                        # Fallback to direct neighbor access
                        for neighbor_id in self._get_level_neighbors(current_node, 0, nodes_dict):
                            admit(neighbor_id, None)
                        continue

                    # Only access cluster once, then cache. A cross edge back
                    # into a cluster we already hold costs nothing: its pointer
                    # simply goes undereferenced.
                    cluster_data = node_cache.get(current_node)

                    if cluster_data is not None:
                        # Still held: refresh its recency so eviction is LRU
                        cluster_cache.move_to_end(cluster_data.cluster_id)
                    else:
                        if cluster_ptr is None:
                            # An intra-cluster candidate carries no pointer, so
                            # if its cluster was evicted before we popped it we
                            # re-fetch through the pointer that reached that
                            # cluster the first time. Keeping the handle costs a
                            # pointer, not a cluster, so the cap still bounds
                            # what the client holds.
                            cluster_ptr = cluster_ptrs.get(self.node_clusters[current_node])

                        if cluster_ptr is None:
                            # Entry node: the one place left that has to look a
                            # cluster up instead of being handed a pointer.
                            # entry_points is a SmartAVLTree, so this is an
                            # oblivious lookup, not a free dict hit.
                            cluster_ptr = self.entry_points[self.node_clusters[current_node]]

                        # first time accessing this cluster, or first since it
                        # was evicted
                        logical_search_counter.increment_search_round()
                        cluster_data = SmartPointer.get(cluster_ptr)
                        cluster_cache[cluster_data.cluster_id] = cluster_data
                        cluster_cache.move_to_end(cluster_data.cluster_id)
                        cluster_ptrs[cluster_data.cluster_id] = cluster_ptr
                        visited_clusters.add(cluster_data.cluster_id)

                        # Holding the cluster means holding all of its nodes
                        for member_id in cluster_data.nodes:
                            node_cache[member_id] = cluster_data

                        # Evict LRU-first, never the cluster just fetched: it is
                        # MRU above, and is read again right below.
                        while cache_capacity and len(cluster_cache) > cache_capacity:
                            _, evicted = cluster_cache.popitem(last=False)
                            for member_id in evicted.nodes:
                                if node_cache.get(member_id) is evicted:
                                    del node_cache[member_id]

                    # get neighbors using cached cluster data (no counter increment)
                    intra, cross = self.get_neighbors_in_cluster(
                        current_node, cluster_data, with_pointers=True
                    )

                    # Same cluster: already held, no pointer needed
                    for neighbor_id in intra:
                        admit(neighbor_id, None)

                    # Other clusters: the edge carries the way there
                    for neighbor_id, neighbor_cluster_ptr in cross:
                        admit(neighbor_id, neighbor_cluster_ptr)

                # ==================== NO CLUSTERING (Standard HNSW) ====================
                else:
                    # Use standard neighbor retrieval (respects oblivious_upper_layers setting)
                    for neighbor_id in self._get_level_neighbors(current_node, 0, nodes_dict):
                        admit(neighbor_id, None)

            except Exception as e:
                print(f"Error processing node {current_node}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Extract results
        results = []
        while dynamic_candidates:
            neg_dist, node_id = heappop(dynamic_candidates)
            results.append((node_id, -neg_dist))

        results.sort(key=lambda x: x[1])
        return results
    
    # ==================== LOADING AND INITIALIZATION ====================
    
    def load_hnsw_from_json(self, json_path: str, vectors_path: str = None, use_weights=True):
        """Load HNSW and apply the configured clustering/partitioning method."""
        # Load JSON structure
        v_names, edges, meta, nodes_dict = self._load_hnsw_json_to_graph(json_path, use_weights=use_weights)
        
        self.max_level = meta["max_level"]
        self.entry_point = meta["entry_point"]
        ntotal = meta["ntotal"]
        
        # Load vectors (required for K-means)
        if vectors_path:
            self._load_vectors(vectors_path)
        
        # Extract base layer edges
        base_edges = self._build_base_layer_graph()
        
        # Apply clustering if enabled
        if self.clustering_method == 'kmeans':
            print("Applying K-means clustering to base layer...")
            self._emulate_graph_clustered_partitioned(base_edges, ntotal)
        elif self.clustering_method == 'metis':
            print("Applying METIS graph partitioning to base layer...")
            self._emulate_graph_clustered_partitioned(base_edges, ntotal)
        elif self.clustering_method == 'singleton':
            print("Applying singleton partitioning to base layer (one node per partition)...")
            self._emulate_graph_clustered_partitioned(base_edges, ntotal)
        elif self.clustering_method == 'louvain':
            print("Applying graph clustering (Louvain)...")
            self.emulate_graph(edges=edges, move_semantics=True, use_clustering=True, sort_names=True)
        else:
            print("No clustering applied")
            self.emulate_graph(edges=edges, move_semantics=True, sort_names=True)

        print(f"Loaded oblivious HNSW: {ntotal} nodes, {self.max_level + 1} levels")
        return (meta, nodes_dict)
    
    def _load_hnsw_json_to_graph(self, json_path: str, use_weights: bool = True):
        """Load HNSW JSON structure."""
        with open(json_path, "r") as f:
            j = json.load(f)

        meta = j["meta"]
        graph = j["graph"]

        ntotal = int(meta["ntotal"])
        max_level = int(meta["max_level"])
        entry_point = int(meta["entry_point"])

        v_names = {i: i for i in range(ntotal)}
        adjacency = [[[] for _ in range(ntotal)] for _ in range(max_level + 1)]
        
        # Maps vertex ids to sets of neighbor ids
        # Used to ensure add_neighbors / add_neighbors_clustered
        # return the correct results
        # Requires that sort_names=True so vertex names == vertex ids
        nodes_dict = dict() 

        for node_obj in graph:
            node_id = int(node_obj["node"])
            all_neighbors = set()
            for level_obj in node_obj["levels"]:
                level = int(level_obj["layer"])
                neighbors = [int(x) for x in level_obj["neighbors"]]
                adjacency[level][node_id] = neighbors
                all_neighbors.update(neighbors)
            nodes_dict[node_id] = all_neighbors

        # Store level graphs
        self.level_graphs = {}
        for level in range(max_level + 1):
            self.level_graphs[level] = {}
            for node_id in range(ntotal):
                self.level_graphs[level][node_id] = adjacency[level][node_id]

        # Build edge list
        edges = []
        if use_weights:
            for level in range(max_level + 1):
                for u in range(ntotal):
                    for v in adjacency[level][u]:
                        edges.append((u, v, level))
        else:
            for u in range(ntotal):
                for v in adjacency[0][u]:
                    edges.append((u, v))

        meta_out = {
            "ntotal": ntotal,
            "max_level": max_level,
            "entry_point": entry_point,
        }
        
        return (v_names, edges, meta_out, nodes_dict)
    
    def _load_vectors(self, vectors_path: str):
        """Load vectors from .fvecs format."""
        try:
            X, n, d = load_fvecs(vectors_path)
            
            for i in range(n):
                self.vectors[i] = X[i]
            
            print(f"Loaded {n} vectors of dimension {d}")
            
        except Exception as e:
            print(f"Could not load vectors: {e}")
    
    def _distance(self, vec1, vec2, metric="l2"):
        """Calculate distance between vectors."""
        if metric == "l2":
            if np is not None and hasattr(vec1, 'dot'):
                diff = vec1 - vec2
                return float(diff.dot(diff))
            else:
                return float(sum((x - y) * (x - y) for x, y in zip(vec1, vec2)))
        elif metric == "ip":
            if np is not None and hasattr(vec1, 'dot'):
                return float(-float(vec1.dot(vec2)))
            else:
                return float(-sum(x * y for x, y in zip(vec1, vec2)))
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    # ==================== DEBUG UTILITY ====================
    
    def print_access_mode_summary(self):
        """Print a summary of how neighbors will be accessed at each layer."""
        print(f"\n{'='*70}")
        print(f"{'NEIGHBOR ACCESS MODE SUMMARY':^70}")
        print(f"{'='*70}")
        print(f"Clustering method: {self.clustering_method or 'None'}")
        print(f"Oblivious upper layers: {self.oblivious_upper_layers}")
        print(f"{'-'*70}")
        print(f"{'Layer':<10} {'Access Mode':<30} {'Description'}")
        print(f"{'-'*70}")
        
        # Base layer (level 0)
        if self.clustering_method == 'kmeans':
            print(f"{'0 (base)':<10} {'K-means clustering':<30} {'Oblivious (cluster pointer hop)'}")
        elif self.clustering_method == 'metis':
            print(f"{'0 (base)':<10} {'METIS partitioning':<30} {'Oblivious (cluster pointer hop)'}")
        elif self.clustering_method == 'singleton':
            print(f"{'0 (base)':<10} {'Singleton partitions':<30} {'Oblivious (node pointer hop)'}")
        elif self.use_clustering:
            print(f"{'0 (base)':<10} {'Graph clustering':<30} {'Oblivious (Louvain)'}")
        elif self.oblivious_upper_layers:
            print(f"{'0 (base)':<10} {'Standard oblivious':<30} {'Oblivious (OSAM)'}")
        else:
            print(f"{'0 (base)':<10} {'Direct access':<30} {'Non-oblivious'}")
        
        # Upper layers
        for level in range(1, self.max_level + 1):
            if self.use_node_clustering:
                # K-means/METIS: upper layers always direct (not in OSAM structure)
                print(f"{level:<10} {'Direct access (forced)':<30} "
                      f"{'Non-oblivious (' + self.clustering_method + ' only base)'}")
            elif self.oblivious_upper_layers:
                print(f"{level:<10} {'Standard oblivious':<30} {'Oblivious (OSAM)'}")
            else:
                print(f"{level:<10} {'Direct access':<30} {'Non-oblivious'}")
        
        print(f"{'='*70}\n")
    
    
    # ====================  Faiss search and Verification ====================
    
    def load_faiss_index(self, index_path: str):
        """Load pre-built FAISS index."""
        if not _FAISS_AVAILABLE:
            print("FAISS not available")
            return False
            
        try:
            self.faiss_index = faiss.read_index(index_path)
            print(f"Loaded FAISS index: {self.faiss_index.ntotal} vectors")
            
            if hasattr(self.faiss_index, 'hnsw'):
                self.faiss_index.hnsw.efSearch = 64
            
            return True
        except Exception as e:
            print(f"Could not load FAISS index: {e}")
            return False
    
    def load_query_vectors(self, query_path: str):
        """Load query vectors."""
        try:
            Q, nq, d = load_fvecs(query_path)
            query_vectors = [Q[i] for i in range(nq)]
            print(f"Loaded {nq} query vectors")
            return query_vectors
        except Exception as e:
            print(f"Could not load queries: {e}")
            return []
    
    def faiss_search(self, query_vector, k=10, ef_search=None):
        """FAISS search for verification."""
        if not _FAISS_AVAILABLE or self.faiss_index is None:
            return []
        
        try:
            original_ef = None
            if ef_search is not None and hasattr(self.faiss_index, 'hnsw'):
                original_ef = self.faiss_index.hnsw.efSearch
                self.faiss_index.hnsw.efSearch = ef_search
            
            if not isinstance(query_vector, np.ndarray):
                query_np = np.array([query_vector], dtype=np.float32)
            else:
                query_np = query_vector.reshape(1, -1).astype(np.float32)
            
            distances, indices = self.faiss_index.search(query_np, k)
            
            if original_ef is not None and hasattr(self.faiss_index, 'hnsw'):
                self.faiss_index.hnsw.efSearch = original_ef
            
            results = []
            for i in range(len(indices[0])):
                if indices[0][i] != -1:
                    results.append((int(indices[0][i]), float(distances[0][i])))
            
            return results
        except Exception as e:
            print(f"FAISS search error: {e}")
            return []
    
    def verify_search_results(self, query_vectors, ground_truth, nodes_dict, k=10, ef_search=64, num_queries=None, metric="ip", gamma=1):
        """Verify oblivious search against FAISS with comprehensive memory access tracking."""
        if not query_vectors:
            print("No query vectors")
            return

        if num_queries is not None:
            query_vectors = query_vectors[:num_queries]
            ground_truth = ground_truth[:num_queries]
        
        # Validate ground truth matches queries
        if len(query_vectors) != len(ground_truth):
            print(f"Warning: Query count ({len(query_vectors)}) != Ground truth count ({len(ground_truth)})")
            min_count = min(len(query_vectors), len(ground_truth))
            query_vectors = query_vectors[:min_count]
            ground_truth = ground_truth[:min_count]
        
        print(f"\n=== Verification ({len(query_vectors)} queries, k={k}, ef={ef_search}) ===")
        print(f"Clustering: {self.clustering_method or 'None'}")
        print(f"Upper layer mode: {'Oblivious' if self.oblivious_upper_layers else 'Non-oblivious'}")
        
        # Performance tracking arrays
        faiss_times = []
        oblivious_times = []
        recall_scores = []
        all_recall_metrics = []
        
        # Memory access tracking per query
        per_query_reads = []
        per_query_writes = []
        per_query_allocs = []
        per_query_ch_misses = []
        
        # Get initial global counters
        a_start, r_start, w_start = get_global_counter()
        
        # Reset logical search counter
        logical_search_counter.reset()
        
        print("\n=== Running Queries ===")
        for i, query in enumerate(query_vectors):
            print(f"\rQuery {i+1}/{len(query_vectors)}", end="", flush=True)
            
            # Get counters before this query
            a_q_start, r_q_start, w_q_start = get_global_counter()
            
            # FAISS search (for verification)
            faiss_result = []
            if _FAISS_AVAILABLE and self.faiss_index:
                start = time.time()
                faiss_result = self.faiss_search(query, k=k, ef_search=ef_search)
                faiss_times.append(time.time() - start)
            
            # Oblivious HNSW search with memory tracking
            start = time.time()
            oblivious_result = self.oblivious_hnsw_search(query, nodes_dict, k=k, ef_search=ef_search, metric=metric)
            oblivious_times.append(time.time() - start)
            
            # Get counters after this query
            a_q_end, r_q_end, w_q_end = get_global_counter()
            
            # Calculate per-query statistics
            query_reads = r_q_end - r_q_start
            query_writes = w_q_end - w_q_start
            query_allocs = a_q_end - a_q_start
            # query_ch_misses = m_q_end - m_q_start
            
            per_query_reads.append(query_reads)
            per_query_writes.append(query_writes)
            per_query_allocs.append(query_allocs)
            # per_query_ch_misses.append(query_ch_misses)
            
            # Calculate recall
            if faiss_result and oblivious_result:
                faiss_ids = set([x[0] for x in faiss_result])
                oblivious_ids = set([x[0] for x in oblivious_result])
                recall = len(faiss_ids & oblivious_ids) / len(faiss_ids)
                recall_scores.append(recall)
            
            # Calculate recall metrics using pre-loaded ground truth
            if ground_truth[i] and oblivious_result:
                recall_metrics = compute_recall_metrics(
                    ground_truth[i],      # ← Ground truth IDs from file
                    oblivious_result,     # ← O-HNSW results
                    k=k, 
                    gamma=gamma
                )
                all_recall_metrics.append(recall_metrics)
        
        print()  # New line after progress
        
        # Get final global counters
        a_end, r_end, w_end = get_global_counter()
        
        # Calculate total statistics
        total_allocs = a_end - a_start
        total_reads = r_end - r_start
        # total_cache_hits = ch_end - ch_start
        # total_cache_misses = m_end - m_start
        total_writes = w_end - w_start
        
        # Get logical search statistics
        logical_stats = logical_search_counter.get_stats()

        # Calculate average recall
        avg_recall_metrics = compute_average_recall(all_recall_metrics)
        
        # ==================== PRINT DETAILED RESULTS ====================
        
        print(f"\n{'='*60}")
        print(f"{'PERFORMANCE RESULTS':^60}")
        print(f"{'='*60}")
        
        # Timing results
        if faiss_times:
            avg_faiss = sum(faiss_times) / len(faiss_times)
            print(f"FAISS average time:     {avg_faiss:.6f}s")
            print(f"FAISS total time:       {sum(faiss_times):.6f}s")
        else:
            avg_faiss = None
            print("FAISS search:           Not available")
        
        if oblivious_times:
            avg_oblivious = sum(oblivious_times) / len(oblivious_times)
            print(f"Oblivious average time: {avg_oblivious:.6f}s")
            print(f"Oblivious total time:   {sum(oblivious_times):.6f}s")
        else:
            avg_oblivious = None
            print("Oblivious search:       Failed")
        
        if avg_faiss and avg_oblivious and avg_faiss > 0:
            overhead = avg_oblivious / avg_faiss
            print(f"Time overhead:          {overhead:.2f}x")
        
        # Recall results
        # if recall_scores:
        #     avg_recall = sum(recall_scores) / len(recall_scores)
        #     min_recall = min(recall_scores)
        #     max_recall = max(recall_scores)
        #     print(f"\nRecall@{k}:")
        #     print(f"  Average:              {avg_recall:.4f}")
        #     print(f"  Min:                  {min_recall:.4f}")
        #     print(f"  Max:                  {max_recall:.4f}")
        # else:
        #     print("\nRecall:                 Not calculated")
           # Enhanced recall results
        if all_recall_metrics:
            print(f"\n{'*'*60}")
            print(f"{'RECALL METRICS (vs Ground Truth)':^60}")
            print(f"{'*'*60}")
            print(f"R@1  = {avg_recall_metrics['R@1']:.4f}")
            print(f"R@10 = {avg_recall_metrics['R@10']:.4f}")
            if k >= 100:
                print(f"R@100 = {avg_recall_metrics['R@100']:.4f}")
            print(f"(Results for {avg_recall_metrics['num_queries']} queries, k={k}, gamma={gamma})")
        else:
            print("\nRecall:                 Not calculated")
        
        print(f"\n{'='*60}")
        print(f"{'MEMORY ACCESS STATISTICS (TOTAL)':^60}")
        print(f"{'='*60}")
        
        print(f"Total reads:            {total_reads:,}")
        print(f"Total writes:           {total_writes:,}")
        print(f"Total allocations:      {total_allocs:,}")
        # ORAM memory size for bandwidth accounting: allocs over the whole run,
        # dominated by the build phase (queries only add a_end - a_start).
        print(f"Build-phase allocations:{a_start:,}")
        print(f"Whole-run allocations:  {a_end:,}")
        print(f"log2(whole-run allocs): {math.log2(a_end) if a_end > 0 else 0:.2f}")
        # print(f"Total read misses:      {total_read_misses:,}")
        # print(f"Total cache hits:       {total_cache_hits:,}")
        # print(f"Total evictions:        {total_evictions:,}")
        # print(f"Total flushes:          {total_flushes:,}")
        
        # Calculate cache hit rate
        # if total_reads > 0:
            # cache_hit_rate = (total_cache_hits / total_reads) * 100
            # print(f"Cache hit rate:         {cache_hit_rate:.2f}%")
            # print(f"Cache hits:             {(total_cache_hits/num_queries):.2f}")

        # calculate the cache miss rate
        # if total_reads > 0:
            # cache_miss_rate = (total_cache_misses / total_reads) * 100
            # print(f"Cache miss rate:        {cache_miss_rate:.2f}%")
            # print(f"Cache misses:           {(total_cache_misses/num_queries):.2f}")
            # print(f"Cache misses2:           {(sum(per_query_ch_misses)/len(per_query_ch_misses)):.2f}")
        
        print(f"\n{'='*60}")
        print(f"{'MEMORY ACCESS STATISTICS (PER QUERY)':^60}")
        print(f"{'='*60}")
        
        num_queries = len(query_vectors)
        
        # Average per query
        avg_allocs = total_allocs / num_queries
        avg_reads = total_reads / num_queries
        avg_writes = total_writes / num_queries
        
        print(f"Average reads/query:    {avg_reads:.1f}")
        print(f"Average writes/query:   {avg_writes:.1f}")
        print(f"Average allocs/query:   {avg_allocs:.1f}")
        
        # Min/Max per query
        if per_query_reads:
            print(f"\nPer-query ranges:")
            print(f"  Accesses: {min(per_query_allocs):,} - {max(per_query_allocs):,}")
            print(f"  Reads:    {min(per_query_reads):,} - {max(per_query_reads):,}")
            print(f"  Writes:   {min(per_query_writes):,} - {max(per_query_writes):,}")
        
        print(f"\n{'='*60}")
        print(f"{'LOGICAL ROUND TRIP STATISTICS (SEARCH ONLY)':^60}")
        print(f"{'='*60}")
        
        print(f"Total logical rounds:   {logical_stats['total_rounds']:,}")
        print(f"Average rounds/query:   {logical_stats['avg_per_query']:.1f}")
        print(f"Max rounds (1 query):   {logical_stats['max_per_query']}")
        print(f"Min rounds (1 query):   {logical_stats['min_per_query']}")
        print(f"Number of queries:      {logical_stats['num_queries']}")
        
        print(f"\n{'='*60}")
        print(f"{'CONFIGURATION':^60}")
        print(f"{'='*60}")
        
        print(f"Clustering method:      {self.clustering_method or 'None'}")
        print(f"Upper layer mode:       {'Oblivious' if self.oblivious_upper_layers else 'Non-oblivious'}")
        print(f"Queries processed:      {num_queries}")
        print(f"k (neighbors):          {k}")
        print(f"ef_search:              {ef_search}")
        
        # Clustering-specific info
        if self.use_node_clustering:
            if hasattr(self, 'cluster_nodes') and self.cluster_nodes:
                n_clusters = len(self.cluster_nodes)
                avg_cluster_size = sum(len(nodes) for nodes in self.cluster_nodes.values()) / n_clusters
                label = {'metis': 'METIS Partitioning',
                         'singleton': 'Singleton Partitioning'}.get(
                             self.clustering_method, 'K-means Clustering')
                print(f"\n{label}:")
                print(f" Number of clusters:   {n_clusters}")
                print(f" Avg cluster size:     {avg_cluster_size:.1f} nodes")
                print(f" max cluster size:     {max(len(nodes) for nodes in self.cluster_nodes.values())} nodes")
                if self.cluster_counts:
                    # One reference per cluster is its entry point; the rest are
                    # the cross-cluster edges pointing at it
                    copies = sum(self.cluster_counts.values()) - len(self.cluster_counts)
                    print(f" Cross-edge pointers:  {copies}")
                    print(f" Max copies / cluster: {max(self.cluster_counts.values()) - 1}")
        
        # print(f"\n{'='*60}")
        # print(f"{'EFFICIENCY METRICS':^60}")
        # print(f"{'='*60}")
        
        # Calculate efficiency metrics
        # if logical_stats['total_rounds'] > 0 and total_allocs > 0:
        #     # Physical accesses per logical round
        #     accesses_per_logical_round = total_allocs / logical_stats['total_rounds']
        #     print(f"Physical accesses per logical round: {accesses_per_logical_round:.1f}")
        
        # if total_allocs > 0:
        #     read_write_ratio = total_reads / total_writes if total_writes > 0 else float('inf')
        #     print(f"Read/Write ratio:       {read_write_ratio:.2f}")
        
        # if avg_oblivious and avg_accesses > 0:
        #     time_per_access = (avg_oblivious / avg_accesses) * 1e6  # microseconds
        #     print(f"Time per access:        {time_per_access:.3f} µs")
        
        # print(f"{'='*60}\n")
        
        # Return comprehensive statistics
        return {
            # Timing
            'avg_faiss_time': avg_faiss,
            'avg_oblivious_time': avg_oblivious,
            'overhead': (avg_oblivious / avg_faiss) if (avg_faiss and avg_oblivious and avg_faiss > 0) else None,
            
            # # Recall
            # 'avg_recall': sum(recall_scores) / len(recall_scores) if recall_scores else None,
            # 'min_recall': min(recall_scores) if recall_scores else None,
            # 'max_recall': max(recall_scores) if recall_scores else None,

            # Recall (enhanced)
            'recall@1': avg_recall_metrics['R@1'],
            'recall@10': avg_recall_metrics['R@10'],
            'recall@100': avg_recall_metrics.get('R@100', 0.0),
            
            # Total memory access
            'total_allocs': total_allocs,
            'total_reads': total_reads,
            'total_writes': total_writes,

            # 'total_cache_hits': total_cache_hits,
            # 'total_evictions': total_evictions,
            # 'total_flushes': total_flushes,
            
            # Per-query memory access
            'avg_reads_per_query': avg_reads,
            'avg_writes_per_query': avg_writes,
            'avg_allocs_per_query': avg_allocs,
            
            # 'min_accesses_per_query': min(per_query_allocs) if per_query_allocs else 0,
            # 'max_accesses_per_query': max(per_query_allocs) if per_query_allocs else 0,
            
            # Logical rounds
            'total_logical_rounds': logical_stats['total_rounds'],
            'avg_logical_rounds_per_query': logical_stats['avg_per_query'],
            'max_logical_rounds': logical_stats['max_per_query'],
            'min_logical_rounds': logical_stats['min_per_query'],
            
            # Configuration
            'oblivious_upper_layers': self.oblivious_upper_layers,
            'clustering_method': self.clustering_method,
            'num_queries': num_queries,
            'k': k,
            'ef_search': ef_search,
            'gamma': gamma,
            
            # Efficiency metrics
            # 'cache_hit_rate': (total_cache_hits / total_reads * 100) if total_reads > 0 else 0,
            # 'read_write_ratio': total_reads / total_writes if total_writes > 0 else float('inf'),
            # 'accesses_per_logical_round': total_allocs / logical_stats['total_rounds'] if logical_stats['total_rounds'] > 0 else 0,
        }


def parse_cluster_cache_spec(value):
    """
    Parse --cluster-cache into a (kind, value) spec.

        off | none | 0  -> ('fixed', 1)      hold only the cluster being read
        all | unbounded -> ('unbounded', None)
        N               -> ('fixed', N)      at most N partitions resident
        P%              -> ('percent', P)    P% of the total partition count

    'off' is capacity 1 rather than 0 because the cluster currently being read
    has to stay resident; 1 is the no-caching baseline.
    """
    text = value.strip().lower()

    if text in ('all', 'unbounded'):
        return ('unbounded', None)
    if text in ('off', 'none', '0'):
        return ('fixed', 1)

    if text.endswith('%'):
        try:
            percent = float(text[:-1])
        except ValueError:
            raise argparse.ArgumentTypeError(f"invalid percentage: {value!r}")
        if not 0 < percent <= 100:
            raise argparse.ArgumentTypeError(f"percentage must be in (0, 100]: {value!r}")
        return ('percent', percent)

    try:
        count_ = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected 'off', 'all', a positive integer, or a percentage like '25%', got {value!r}"
        )
    if count_ < 1:
        raise argparse.ArgumentTypeError(f"capacity must be >= 1 (use 'off' for 1): {value!r}")
    return ('fixed', count_)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Oblivious HNSW with optional clustering/partitioning')
    parser.add_argument('--q', type=int, default=10, help='Number of queries')
    parser.add_argument('--k', type=int, default=10, help='k nearest neighbors')
    parser.add_argument('--ef', type=int, default=64, help='ef_search')
    parser.add_argument('--gamma', type=int, default=1, help='Recall tolerance factor')
    parser.add_argument('--graph', type=str, default='graph.json', help='Graph JSON')
    parser.add_argument('--vectors', type=str, default='data/1k/laion_base.fvecs', help='Base vectors')
    parser.add_argument('--queries', type=str, default='data/1k/laion_query.fvecs', help='Query vectors')
    parser.add_argument('--gt', type=str, default='data/1k/gt.ivecs', help='Ground truth file (.ivecs)')
    parser.add_argument('--faiss-index', type=str, default='data/1k/hnsw_32_40_ip.index', help='FAISS index')
    parser.add_argument('--non-oblivious-upper', action='store_true', help='Non-oblivious upper layers')
    parser.add_argument('--branching-factor', type=int, default=2, help='Out-tree branching factor (OGraph fan-out)')
    parser.add_argument('--pointer-type', type=str, default='multiwrite',
                       choices=['original', 'multiwrite', 'multiwriterary', 'recursive'],
                       help='SmartPointer implementation to use (default: multiwrite)')
    parser.add_argument('--bs', type=int, default=64,
                       help='Block size in bytes; the multiwriterary pointer branching factor is derived from it')
    parser.add_argument('--pointer-branching-factor', type=int, default=None,
                       help='Multiwriterary pointer branching factor (default: derived from --bs)')
    parser.add_argument('--path-oram-rary-read-evictions', type=int, default=None,
                       help='Full PathORAM evictions to run after each r-ary pointer read '
                            '(default: pointer branching factor)')
    parser.add_argument('--track-stash', action='store_true',
                       help='Replay the PathORAM trace and print stash stats for build/query phases')
    parser.add_argument('--track-stack-size', dest='track_stash', action='store_true',
                       help=argparse.SUPPRESS)
    parser.add_argument('--path-oram-tree-size', type=int, default=None,
                       help='Number of PathORAM leaves for stash replay')
    parser.add_argument('--path-oram-bucket-capacity', type=int, default=4,
                       help='PathORAM bucket capacity for stash replay')
    parser.add_argument('--partitioning', type=str, default=None,
                       choices=['none', 'kmeans', 'metis', 'louvain', 'singleton'],
                       help='Partitioning/clustering method for the base layer (default: none)')
    parser.add_argument('--n-clusters', type=int, default=None,
                       help='Number of clusters/partitions for kmeans/metis (default: sqrt(ntotal))')
    parser.add_argument('--balance-cluster-refs', action='store_true',
                       help='Balance each cluster\'s cross-edge pointer reference tree at build '
                            'time, so dereferencing a cross edge costs ~log(C) rather than O(C) '
                            'in the number of edges entering that cluster (default: off)')
    parser.add_argument('--no-batch-cluster-refs', action='store_true',
                       help='Take each cluster\'s cross-edge pointers one incremental copy per '
                            'edge instead of one batched copy per cluster. Batching is the '
                            'default for the multiwrite/multiwriterary pointer types')
    # Legacy flags, kept for backward compatibility with existing scripts
    parser.add_argument('--use-kmeans-clustering', action='store_true', help='Enable K-means clustering')
    parser.add_argument('--use-louvain-clustering', action='store_true', help='Enable Louvain clustering')
    parser.add_argument('--save-results', type=str, default=None, help='Save results to JSON file')
    parser.add_argument('--metric', type=str, default='ip', choices=['ip', 'l2'], help='Distance metric')
    parser.add_argument('--make-copies', action='store_true', help='Toggle copying')

    

    # How many partitions the client may hold at once during a query
    parser.add_argument('--cluster-cache', type=parse_cluster_cache_spec,
                       default=('unbounded', None), metavar='SPEC',
                       help="Client-side partition cache cap: 'off' (hold only the "
                            "current partition), 'all' (unbounded, default), an integer "
                            "(e.g. 8), or a percentage of the partition count (e.g. 25%%)")

    # Add cache arguments
    parser.add_argument('--cache-capacity', type=int, default=100,
                       help='ORAM block cache capacity (0 to disable) -- currently inert, '
                            'set_cache is disabled below')
    parser.add_argument('--cache-eviction', type=float, default=0.2,
                       help='Percentage to evict when ORAM block cache full (0.0-1.0) -- '
                            'currently inert')

    args = parser.parse_args()

    if args.pointer_branching_factor is not None and args.pointer_branching_factor <= 0:
        parser.error("--pointer-branching-factor must be a positive integer")
    if args.path_oram_rary_read_evictions is not None and args.path_oram_rary_read_evictions <= 0:
        parser.error("--path-oram-rary-read-evictions must be a positive integer")
    if args.path_oram_bucket_capacity <= 0:
        parser.error("--path-oram-bucket-capacity must be a positive integer")
    if args.track_stash and args.path_oram_tree_size is None:
        parser.error("--track-stash requires --path-oram-tree-size")

    stash_tracking = args.track_stash or args.path_oram_tree_size is not None
    pointer_branching_factor = (
        args.pointer_branching_factor
        if args.pointer_branching_factor is not None
        else math.floor((args.bs - 2 * 8) / 8)
    )
    rary_read_evictions = (
        args.path_oram_rary_read_evictions
        if args.path_oram_rary_read_evictions is not None
        else pointer_branching_factor
    )

    # Configure the ORAM block cache. set_cache is commented out, so these two
    # arguments currently have no effect on any run -- do not report them as an
    # active setting.
    if args.cache_capacity > 0:
        # from single_access_machine import set_cache
        # set_cache(capacity=args.cache_capacity, percent=args.cache_eviction)
        print(f"! ORAM block cache NOT configured (set_cache disabled); "
              f"--cache-capacity {args.cache_capacity} / --cache-eviction "
              f"{args.cache_eviction*100}% are inert")
    else:
        print("✓ ORAM block cache disabled")

    # --partitioning takes precedence; legacy flags are the fallback
    if args.partitioning is not None:
        clustering_method = None if args.partitioning == 'none' else args.partitioning
    else:
        clustering_method = 'louvain' if args.use_louvain_clustering else None
        if args.use_kmeans_clustering: clustering_method = 'kmeans'

    if clustering_method == 'metis' and not _METIS_AVAILABLE:
        print("✗ METIS partitioning requested but pymetis is not installed.")
        print("  Install with: pip install pymetis (requires libmetis)")
        return

    # Singleton fixes the partition count at ntotal, so there is nothing for
    # --n-clusters to set. Say so rather than accepting it and ignoring it.
    if clustering_method == 'singleton' and args.n_clusters is not None:
        print(f"! --n-clusters {args.n_clusters} ignored: singleton partitioning "
              f"always uses one partition per node")

    title = f"OBLIVIOUS HNSW ({clustering_method or 'no'} partitioning)".upper()
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    print(f"Queries:            {args.q}")
    print(f"k:                  {args.k}")
    print(f"ef_search:          {args.ef}")
    print(f"gamma:              {args.gamma}")
    print(f"distance metric:    {args.metric}")
    print(f"Clustering:         {clustering_method or 'None'}")
    if clustering_method == 'singleton':
        print(f"Num clusters:       ntotal (one node per partition)")
    elif clustering_method in NODE_CLUSTERED_METHODS:
        print(f"Num clusters:       {args.n_clusters if args.n_clusters else 'auto (sqrt(ntotal))'}")
    print(f"Upper layer mode:   {'Non-oblivious' if args.non_oblivious_upper else 'Oblivious'}")
    print(f"Out-tree branching: {args.branching_factor}")
    print(f"Pointer type:       {args.pointer_type}")
    print(f"Pointer block size: {args.bs} bytes")
    if args.pointer_type == 'multiwriterary':
        print(f"Pointer branching:  {pointer_branching_factor}")
        print(f"R-ary read evicts:  {rary_read_evictions}")
    print(f"Stash tracking:     {'Enabled' if stash_tracking else 'Disabled'}")
    if stash_tracking:
        print(f"PathORAM leaves:    {args.path_oram_tree_size}")
        print(f"PathORAM bucket Z:  {args.path_oram_bucket_capacity}")
    _cc_kind, _cc_value = args.cluster_cache
    if _cc_kind == 'unbounded':
        _cc_text = 'unbounded'
    elif _cc_kind == 'percent':
        _cc_text = f"{_cc_value}% of partitions"
    else:
        _cc_text = 'off (1 partition)' if _cc_value == 1 else f"{_cc_value} partitions"
    print(f"Cluster cache:      {_cc_text}")
    print(f"{'='*60}\n")
    
    logical_search_counter.reset()
    if stash_tracking:
        set_path_oram_tree_size(args.path_oram_tree_size)
        set_path_oram_bucket_capacity(args.path_oram_bucket_capacity)
    else:
        set_path_oram_tree_size(None)
    
    # Build oblivious HNSW
    print("=== Building Oblivious HNSW ===")
    SmartPointer.set_smart_pointer(args.pointer_type)
    if args.pointer_type == 'multiwriterary':
        print(f"Setting branching factor for multiwriterary pointer to {pointer_branching_factor}")
        print(f"Setting r-ary read evictions to {rary_read_evictions}")
        SmartPointer.set_branching_factor(pointer_branching_factor)
        set_path_oram_rary_read_evictions(rary_read_evictions)
    else:
        set_path_oram_rary_read_evictions(None)
    o_hnsw = ObliviousHNSW(
        branching_factor=args.branching_factor,
        clustering_method=clustering_method,
        use_osam=True,
        make_copies=args.make_copies,
        n_clusters=args.n_clusters
    )
    o_hnsw.balance_cluster_refs = args.balance_cluster_refs
    o_hnsw.batch_cluster_refs = not args.no_batch_cluster_refs
    o_hnsw.cluster_cache_spec = args.cluster_cache
    o_hnsw.set_oblivious_upper_layers(not args.non_oblivious_upper)
    
    # Load data
    print("\n=== Loading Data ===")
    try:
        meta, nodes_dict = o_hnsw.load_hnsw_from_json(args.graph, args.vectors)
        print(f"✓ Loaded HNSW graph: {meta['ntotal']} nodes, {meta['max_level']+1} levels")
        
        # Print access mode summary
        o_hnsw.print_access_mode_summary()
        if stash_tracking:
            print("\nBuild PathORAM stash stats:")
            print_path_oram_stash_stats()
            reset_write_batches()

        # Partitioning is done, so a percentage cap can now be resolved
        if o_hnsw.use_node_clustering:
            print(f"✓ Cluster cache: {o_hnsw.describe_cluster_cache()}")
        
        faiss_loaded = o_hnsw.load_faiss_index(args.faiss_index)
        if faiss_loaded:
            print(f"✓ Loaded FAISS index")
        else:
            print("✗ FAISS index not available")
        
        query_vectors = o_hnsw.load_query_vectors(args.queries)
        if query_vectors:
            print(f"✓ Loaded {len(query_vectors)} query vectors")
        else:
            print("✗ Could not load query vectors")
            return
        
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return
    
        # ==================== LOAD GROUND TRUTH ====================
    try:
        ground_truth = load_ground_truth(args.gt, num_queries=args.q)
    except FileNotFoundError as e:
        print(f"✗ {e}")
        print("Ground truth is required for recall calculation!")
        return
    except Exception as e:
        print(f"✗ Error loading ground truth: {e}")
        return
    
    # Run verification with ground truth
    print("\n=== Running Verification ===")
    results = o_hnsw.verify_search_results(
        query_vectors, 
        ground_truth,      # ← Pass ground truth
        nodes_dict,
        k=args.k, 
        ef_search=args.ef, 
        num_queries=args.q,
        metric=args.metric,
        gamma=args.gamma
    )
    if stash_tracking:
        print("\nQuery PathORAM stash stats:")
        print_path_oram_stash_stats()
    
    # Save results if requested
    if args.save_results and results:
        try:
            with open(args.save_results, 'w') as f:
                serializable_results = {}
                for key, value in results.items():
                    if value == float('inf'):
                        serializable_results[key] = 'Infinity'
                    elif isinstance(value, (int, float, str, bool, type(None))):
                        serializable_results[key] = value
                    else:
                        serializable_results[key] = str(value)
                
                json.dump(serializable_results, f, indent=2)
            print(f"\n✓ Results saved to {args.save_results}")
        except Exception as e:
            print(f"\n✗ Could not save results: {e}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"{'EXECUTION SUMMARY':^60}")
    print(f"{'='*60}")
    
    if results:
        print(f"✓ Verification completed successfully")
        print(f"  Queries processed:    {results['num_queries']}")
        print(f"  R@1:                  {results['recall@1']:.4f}")
        print(f"  R@10:                 {results['recall@10']:.4f}")
        if args.k >= 100:
            print(f"  R@100:                {results['recall@100']:.4f}")
        print(f"  Logical rounds/query: {results['avg_logical_rounds_per_query']:.1f}")
        print(f"  Memory reads/query:   {results['avg_reads_per_query']:.1f}")
        print(f"  Memory writes/query:  {results['avg_writes_per_query']:.1f}")
    else:
        print(f"✗ Verification failed")
    
    print(f"{'='*60}\n")
    # # Run verification
    # print("\n=== Running Verification ===")
    # results = o_hnsw.verify_search_results(
    #     query_vectors, 
    #     k=args.k, 
    #     ef_search=args.ef, 
    #     num_queries=args.q,
    #     metric=args.metric
    # )
    
    # # Save results if requested
    # if args.save_results and results:
    #     try:
    #         with open(args.save_results, 'w') as f:
    #             # Convert non-serializable values
    #             serializable_results = {}
    #             for key, value in results.items():
    #                 if value == float('inf'):
    #                     serializable_results[key] = 'inf'
    #                 elif isinstance(value, (int, float, str, bool, type(None))):
    #                     serializable_results[key] = value
    #                 else:
    #                     serializable_results[key] = str(value)
                
    #             json.dump(serializable_results, f, indent=2)
    #         print(f"\n✓ Results saved to {args.save_results}")
    #     except Exception as e:
    #         print(f"\n✗ Could not save results: {e}")
    
    # # Print summary
    # print(f"\n{'='*60}")
    # print(f"{'EXECUTION SUMMARY':^60}")
    # print(f"{'='*60}")
    
    # if results:
    #     print(f"✓ Verification completed successfully")
    #     print(f"  Queries processed:    {results['num_queries']}")
    #     print(f"  Average recall:       {results['avg_recall']:.4f}" if results['avg_recall'] else "  Average recall:       N/A")
    #     print(f"  Time overhead:        {results['overhead']:.2f}x" if results['overhead'] else "  Time overhead:        N/A")
    #     print(f"  Logical rounds/query: {results['avg_logical_rounds_per_query']:.1f}")
    #     print(f"  Memory reads/query:  {results['avg_reads_per_query']:.1f}")
    #     print(f"  Memory writes/query:  {results['avg_writes_per_query']:.1f}")
    # else:
    #     print(f"✗ Verification failed")
    
    # print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
