#!/usr/bin/env python3
"""
HNSW Query Hop Counter - Tracks average hops per layer during query traversal
"""

import json
import struct
import argparse
import time
from heapq import heappush, heappop, heapreplace, nlargest
from typing import Dict, List, Tuple
from dataclasses import dataclass, field

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


# ==================== FILE LOADERS ====================

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
class LayerHopStats:
    """Track hop statistics per layer"""
    layer: int
    total_hops: int = 0
    query_count: int = 0
    hops_per_query: List[int] = field(default_factory=list)
    
    @property
    def avg_hops(self) -> float:
        return self.total_hops / self.query_count if self.query_count > 0 else 0.0
    
    @property
    def min_hops(self) -> int:
        return min(self.hops_per_query) if self.hops_per_query else 0
    
    @property
    def max_hops(self) -> int:
        return max(self.hops_per_query) if self.hops_per_query else 0


@dataclass 
class QueryStats:
    """Statistics for a single query"""
    hops_per_layer: Dict[int, int] = field(default_factory=dict)
    total_hops: int = 0
    nodes_visited: int = 0

@dataclass
class LayerGraphStats:
    """Graph structure statistics per layer"""
    layer: int
    num_nodes: int = 0
    num_edges: int = 0
    min_out_degree: int = 0
    max_out_degree: int = 0
    avg_out_degree: float = 0.0
    min_in_degree: int = 0
    max_in_degree: int = 0
    avg_in_degree: float = 0.0

class HopTrackingHNSW:
    """HNSW search with per-layer hop tracking"""
    
    def __init__(self, json_path: str, vectors_path: str, metric: str = "l2"):
        self.metric = metric
        self.vectors = None
        self.adjacency = {}  # layer -> {node_id -> [neighbors]}
        self.max_level = 0
        self.entry_point = 0
        self.ntotal = 0
        
        # Per-layer statistics
        self.layer_stats: Dict[int, LayerHopStats] = {}
        
        # Load data
        self._load_graph(json_path)
        self._load_vectors(vectors_path)
        
        # Initialize layer stats
        for layer in range(self.max_level + 1):
            self.layer_stats[layer] = LayerHopStats(layer=layer)
        
        self._compute_graph_stats()
    
    def _load_graph(self, json_path: str):
        """Load HNSW graph from JSON"""
        print(f"Loading graph from {json_path}...")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        meta = data['meta']
        self.ntotal = int(meta['ntotal'])
        self.max_level = int(meta['max_level'])
        self.entry_point = int(meta['entry_point'])
        
        # Initialize adjacency lists
        for layer in range(self.max_level + 1):
            self.adjacency[layer] = {}
            for i in range(self.ntotal):
                self.adjacency[layer][i] = []
        
        # Populate adjacency lists
        for node_obj in data['graph']:
            node_id = int(node_obj['node'])
            for level_obj in node_obj['levels']:
                layer = int(level_obj['layer'])
                neighbors = [int(x) for x in level_obj['neighbors']]
                self.adjacency[layer][node_id] = neighbors
        
        print(f"  Loaded: {self.ntotal} nodes, {self.max_level + 1} layers, entry_point={self.entry_point}")
    
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
        
        print(f"  Loaded: {len(vectors)} vectors of dimension {len(vectors[0])}")
    
    def _distance(self, vec1, vec2) -> float:
        """Compute distance between vectors"""
        if self.metric == "ip":  # Inner product (negative for min-heap)
            if HAS_NUMPY:
                return -float(np.dot(vec1, vec2))
            else:
                return -sum(a * b for a, b in zip(vec1, vec2))
        else:  # L2
            if HAS_NUMPY:
                diff = vec1 - vec2
                return float(np.dot(diff, diff))
            else:
                return sum((a - b) ** 2 for a, b in zip(vec1, vec2))
    
    def _greedy_search_layer(self, query, layer: int, start_node: int) -> Tuple[int, int]:
        """
        Greedy search at a single layer.
        Returns: (best_node, hop_count)
        """
        current = start_node
        current_dist = self._distance(self.vectors[current], query)
        hops = 0
        
        while True:
            improved = False
            neighbors = self.adjacency[layer].get(current, [])
            
            for neighbor in neighbors:
                hops += 1  # Count each neighbor check as a hop
                neighbor_dist = self._distance(self.vectors[neighbor], query)
                
                if neighbor_dist < current_dist:
                    current = neighbor
                    current_dist = neighbor_dist
                    improved = True
                    break  # Greedy: take first improvement
            
            if not improved:
                break
        
        return current, hops
    
    def _ef_search_layer(self, query, layer: int, entry_node: int, ef: int, k: int) -> Tuple[List[Tuple[int, float]], int]:
        """
        ef-search at base layer.
        Returns: (results, hop_count)
        """
        visited = set([entry_node])
        entry_dist = self._distance(self.vectors[entry_node], query)
        hops = 0
        
        # Min-heap for candidates
        candidates = [(entry_dist, entry_node)]
        # Max-heap for top-ef results (negative distance)
        top_ef = [(-entry_dist, entry_node)]
        
        while candidates:
            cand_dist, cand_node = heappop(candidates)
            
            # Early termination
            worst_dist = -top_ef[0][0]
            if cand_dist > worst_dist:
                break
            
            neighbors = self.adjacency[layer].get(cand_node, [])
            
            for neighbor in neighbors:
                if neighbor in visited:
                    continue
                
                visited.add(neighbor)
                hops += 1  # Count each neighbor visit as a hop
                
                neighbor_dist = self._distance(self.vectors[neighbor], query)
                
                if len(top_ef) < ef:
                    heappush(top_ef, (-neighbor_dist, neighbor))
                    heappush(candidates, (neighbor_dist, neighbor))
                else:
                    if neighbor_dist < -top_ef[0][0]:
                        heapreplace(top_ef, (-neighbor_dist, neighbor))
                        heappush(candidates, (neighbor_dist, neighbor))
        
        # Extract top-k results
        results = []
        while top_ef:
            neg_dist, node = heappop(top_ef)
            results.append((node, -neg_dist))
        results.sort(key=lambda x: x[1])
        
        return results[:k], hops
    
    def search(self, query, k: int = 10, ef: int = 64) -> Tuple[List[Tuple[int, float]], QueryStats]:
        """
        Perform HNSW search with hop tracking.
        Returns: (results, query_stats)
        """
        stats = QueryStats()
        current = self.entry_point
        
        # Phase 1: Greedy search from top layer down to layer 1
        for layer in range(self.max_level, 0, -1):
            current, hops = self._greedy_search_layer(query, layer, current)
            stats.hops_per_layer[layer] = hops
            stats.total_hops += hops
        
        # Phase 2: ef-search at base layer (layer 0)
        results, hops = self._ef_search_layer(query, 0, current, ef, k)
        stats.hops_per_layer[0] = hops
        stats.total_hops += hops
        
        return results, stats
    
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
            'max': max(recalls) if recalls else 0,
            'all': recalls
        }
    
    def run_queries(self, queries_path: str, num_queries: int = 100, k: int = 10, ef: int = 64,
                    gt_path: str = None):
        """Run queries and collect statistics"""
        # Load query vectors
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
            try:
                ground_truth = load_ivecs(gt_path)
                print(f"  Loaded ground truth: {len(ground_truth)} queries")
            except FileNotFoundError:
                print(f"  Ground truth not found: {gt_path}")
        
        num_queries = min(num_queries, len(queries))
        if ground_truth:
            num_queries = min(num_queries, len(ground_truth))
        print(f"  Running {num_queries} queries with k={k}, ef={ef}")
        
        # Reset statistics
        for layer in self.layer_stats.values():
            layer.total_hops = 0
            layer.query_count = 0
            layer.hops_per_query = []
        
        # Store all results for recall computation
        all_results = []
        
        # Run queries
        start_time = time.time()
        for i in range(num_queries):
            if (i + 1) % 100 == 0 or i == num_queries - 1:
                print(f"\r  Progress: {i + 1}/{num_queries}", end="", flush=True)
            
            query = queries[i]
            results, stats = self.search(query, k=k, ef=ef)
            all_results.append(results)
            
            # Update layer statistics
            for layer, hops in stats.hops_per_layer.items():
                self.layer_stats[layer].total_hops += hops
                self.layer_stats[layer].query_count += 1
                self.layer_stats[layer].hops_per_query.append(hops)
        
        elapsed = time.time() - start_time
        print(f"\n  Completed in {elapsed:.2f}s ({num_queries/elapsed:.1f} QPS)")
        
        # Store timing info
        self.query_time = elapsed
        self.num_queries_run = num_queries
        self.k_value = k
        
        # Compute recall if ground truth available
        self.recall_metrics = None
        if ground_truth:
            self.recall_metrics = self.compute_recall(all_results, ground_truth[:num_queries], k)
            print(f"  Recall@{k}: {self.recall_metrics['avg']:.4f}")
        
        return num_queries
    
    def _compute_graph_stats(self):
        """Compute graph structure statistics per layer"""
        from collections import defaultdict
        
        self.graph_stats: Dict[int, LayerGraphStats] = {}
        
        for layer in range(self.max_level + 1):
            # Count nodes with at least one neighbor in this layer
            nodes_in_layer = []
            out_degrees = []
            in_degree_count = defaultdict(int)
            total_edges = 0
            
            for node_id, neighbors in self.adjacency[layer].items():
                if len(neighbors) > 0:
                    nodes_in_layer.append(node_id)
                    out_degrees.append(len(neighbors))
                    total_edges += len(neighbors)
                    
                    # Count in-degrees
                    for neighbor in neighbors:
                        in_degree_count[neighbor] += 1
            
            num_nodes = len(nodes_in_layer)
            
            if num_nodes > 0:
                # Out-degree stats
                min_out = min(out_degrees)
                max_out = max(out_degrees)
                avg_out = sum(out_degrees) / num_nodes
                
                # In-degree stats (only for nodes present in this layer)
                in_degrees = [in_degree_count[n] for n in nodes_in_layer]
                min_in = min(in_degrees) if in_degrees else 0
                max_in = max(in_degrees) if in_degrees else 0
                avg_in = sum(in_degrees) / len(in_degrees) if in_degrees else 0.0
            else:
                min_out = max_out = min_in = max_in = 0
                avg_out = avg_in = 0.0
            
            self.graph_stats[layer] = LayerGraphStats(
                layer=layer,
                num_nodes=num_nodes,
                num_edges=total_edges,
                min_out_degree=min_out,
                max_out_degree=max_out,
                avg_out_degree=avg_out,
                min_in_degree=min_in,
                max_in_degree=max_in,
                avg_in_degree=avg_in
            )

    def print_graph_structure(self, dataset_name: str = ""):
        """Print graph structure statistics per layer"""
        print(f"\n{'='*100}")
        print(f"GRAPH STRUCTURE: {dataset_name}" if dataset_name else "GRAPH STRUCTURE")
        print(f"{'='*100}")
        print(f"Dataset size: {self.ntotal:,} total nodes")
        print(f"Number of layers: {self.max_level + 1}")
        print(f"{'-'*100}")
        print(f"{'Layer':<8} {'Nodes':>10} {'Edges':>12} {'Out-Degree':<30} {'In-Degree':<30}")
        print(f"{'':8} {'':>10} {'':>12} {'Min':>8} {'Max':>8} {'Avg':>10} {'Min':>8} {'Max':>8} {'Avg':>10}")
        print(f"{'-'*100}")
        
        total_edges = 0
        for layer in range(self.max_level, -1, -1):
            gs = self.graph_stats[layer]
            total_edges += gs.num_edges
            print(f"L{layer:<7} {gs.num_nodes:>10,} {gs.num_edges:>12,} "
                f"{gs.min_out_degree:>8} {gs.max_out_degree:>8} {gs.avg_out_degree:>10.2f} "
                f"{gs.min_in_degree:>8} {gs.max_in_degree:>8} {gs.avg_in_degree:>10.2f}")
        
        print(f"{'-'*100}")
        print(f"{'TOTAL':<8} {'':>10} {total_edges:>12,}")
        print(f"{'='*100}\n")
        
        return {
            'total_edges': total_edges,
            'layer_graph_stats': {
                layer: {
                    'num_nodes': gs.num_nodes,
                    'num_edges': gs.num_edges,
                    'min_out_degree': gs.min_out_degree,
                    'max_out_degree': gs.max_out_degree,
                    'avg_out_degree': gs.avg_out_degree,
                    'min_in_degree': gs.min_in_degree,
                    'max_in_degree': gs.max_in_degree,
                    'avg_in_degree': gs.avg_in_degree,
                }
                for layer, gs in self.graph_stats.items()
            }
        }
    
    def print_statistics(self, dataset_name: str = ""):
        """Print hop statistics per layer"""
        # First print graph structure
        graph_info = self.print_graph_structure(dataset_name)

        # print(f"Graph structure: {graph_info}")
        print(f"\n{'='*70}")
        print(f"HOP STATISTICS: {dataset_name}" if dataset_name else "HOP STATISTICS")
        print(f"{'='*70}")
        print(f"Dataset size: {self.ntotal:,} nodes")
        print(f"Number of layers: {self.max_level + 1}")
        print(f"Entry point: {self.entry_point}")
        print(f"{'-'*70}")
        print(f"{'Layer':<8} {'Avg Hops':<12} {'Min Hops':<10} {'Max Hops':<10} {'Total Hops':<12} {'Queries':<8}")
        print(f"{'-'*70}")
        
        total_hops = 0
        for layer in range(self.max_level, -1, -1):
            stats = self.layer_stats[layer]
            total_hops += stats.total_hops
            print(f"L{layer:<7} {stats.avg_hops:<12.2f} {stats.min_hops:<10} {stats.max_hops:<10} {stats.total_hops:<12,} {stats.query_count:<8}")
        
        print(f"{'-'*70}")
        num_queries = self.layer_stats[0].query_count
        avg_total = total_hops / num_queries if num_queries > 0 else 0
        print(f"{'TOTAL':<8} {avg_total:<12.2f} {'':<10} {'':<10} {total_hops:<12,} {num_queries:<8}")
        
        # Recall and speed statistics
        print(f"\n{'-'*70}")
        print(f"RECALL & SPEED STATISTICS")
        print(f"{'-'*70}")
        if hasattr(self, 'recall_metrics') and self.recall_metrics:
            k = getattr(self, 'k_value', 10)
            print(f"Recall@{k}: {self.recall_metrics['avg']:.4f} "
                  f"(min: {self.recall_metrics['min']:.4f}, max: {self.recall_metrics['max']:.4f})")
        else:
            print("Recall: N/A (no ground truth provided)")
        if hasattr(self, 'query_time') and hasattr(self, 'num_queries_run'):
            qps = self.num_queries_run / self.query_time
            latency_ms = (self.query_time / self.num_queries_run) * 1000
            print(f"Speed: {qps:.2f} QPS, {latency_ms:.3f} ms/query")
            print(f"Total time: {self.query_time:.4f}s for {self.num_queries_run} queries")
        print(f"{'='*70}\n")
        
        return {
            'dataset_size': self.ntotal,
            'num_layers': self.max_level + 1,
            'num_queries': num_queries,
            'total_avg_hops': avg_total,
            'layer_stats': {
                layer: {
                    'avg_hops': stats.avg_hops,
                    'min_hops': stats.min_hops,
                    'max_hops': stats.max_hops,
                }
                for layer, stats in self.layer_stats.items()
            },
            'recall': self.recall_metrics['avg'] if hasattr(self, 'recall_metrics') and self.recall_metrics else None,
            'qps': self.num_queries_run / self.query_time if hasattr(self, 'query_time') and self.query_time > 0 else None,
        }


def main():
    parser = argparse.ArgumentParser(description='HNSW Query Hop Counter')
    parser.add_argument('--graph', type=str, required=True, help='Path to HNSW JSON graph')
    parser.add_argument('--vectors', type=str, required=True, help='Path to base vectors (.fvecs)')
    parser.add_argument('--queries', type=str, required=True, help='Path to query vectors (.fvecs)')
    parser.add_argument('--gt', type=str, default=None, help='Path to ground truth (.ivecs)')
    parser.add_argument('--num-queries', type=int, default=100, help='Number of queries to run')
    parser.add_argument('--k', type=int, default=10, help='Number of nearest neighbors')
    parser.add_argument('--ef', type=int, default=64, help='ef_search parameter')
    parser.add_argument('--metric', type=str, default='l2', choices=['ip', 'l2'], help='Distance metric')
    parser.add_argument('--name', type=str, default='', help='Dataset name for output')
    
    args = parser.parse_args()
    
    # Create HNSW instance
    hnsw = HopTrackingHNSW(
        json_path=args.graph,
        vectors_path=args.vectors,
        metric=args.metric
    )
    
    # Run queries
    hnsw.run_queries(
        queries_path=args.queries,
        num_queries=args.num_queries,
        k=args.k,
        ef=args.ef,
        gt_path=args.gt
    )
    
    # Print statistics
    hnsw.print_statistics(dataset_name=args.name)


def run_all_datasets():
    """Convenience function to run all datasets"""
    datasets = [
        {
            'name': '1k',
            'graph': 'data/1k/hnsw_32_40.json',
            'vectors': 'data/1k/laion_base.fvecs',
            'queries': 'data/1k/laion_query.fvecs',
            'gt': 'data/1k/gt.ivecs',
        },
        {
            'name': '10k',
            'graph': 'data/10k/hnsw_32_40.json',
            'vectors': 'data/10k/laion_base.fvecs',
            'queries': 'data/laion_query.fvecs',
            'gt': 'data/10k/gt.ivecs',
        },
        {
            'name': '100k',
            'graph': 'data/100k/hnsw_32_40.json',
            'vectors': 'data/100k/laion_base.fvecs',
            'queries': 'data/laion_query.fvecs',
            'gt': 'data/100k/gt.ivecs',
        },
        {
            'name': '1M',
            'graph': 'data/1m/hnsw_32_40.json',
            'vectors': 'data/1m/laion_base.fvecs',
            'queries': 'data/laion_query.fvecs',
            'gt': 'data/1m/gt.ivecs',
        },
    ]
    
    all_results = {}
    all_recalls = {}
    
    for ds in datasets:
        print(f"\n{'#'*70}")
        print(f"# Processing Dataset: {ds['name']}")
        print(f"{'#'*70}")
        
        try:
            hnsw = HopTrackingHNSW(
                json_path=ds['graph'],
                vectors_path=ds['vectors'],
                metric='l2'
            )
            
            hnsw.run_queries(
                queries_path=ds['queries'],
                num_queries=100,
                k=10,
                ef=64,
                gt_path=ds.get('gt')
            )
            
            results = hnsw.print_statistics(dataset_name=ds['name'])
            all_results[ds['name']] = results
            
            # Store recall and speed info
            all_recalls[ds['name']] = {
                'recall': hnsw.recall_metrics['avg'] if hnsw.recall_metrics else None,
                'qps': hnsw.num_queries_run / hnsw.query_time if hasattr(hnsw, 'query_time') else None
            }
            
        except Exception as e:
            print(f"Error processing {ds['name']}: {e}")
    
    # Print comparison table
    print(f"\n{'='*100}")
    print(f"{'COMPARISON SUMMARY':^100}")
    print(f"{'='*100}")
    print(f"{'Dataset':<10} {'Size':>12} {'Layers':>8} {'Avg Hops':>12} {'L0 Avg':>10} {'Upper Avg':>12} {'Recall@10':>12} {'QPS':>10}")
    print(f"{'-'*100}")
    
    for name, res in all_results.items():
        upper_layer_hops = sum(
            res['layer_stats'][l]['avg_hops'] 
            for l in range(1, res['num_layers'])
        )
        recall_val = all_recalls.get(name, {}).get('recall')
        qps_val = all_recalls.get(name, {}).get('qps')
        recall_str = f"{recall_val:.4f}" if recall_val is not None else "N/A"
        qps_str = f"{qps_val:.1f}" if qps_val is not None else "N/A"
        print(f"{name:<10} {res['dataset_size']:>12,} {res['num_layers']:>8} {res['total_avg_hops']:>12.2f} "
              f"{res['layer_stats'][0]['avg_hops']:>10.2f} {upper_layer_hops:>12.2f} {recall_str:>12} {qps_str:>10}")
    
    print(f"{'='*100}\n")

    # Graph structure summary
    print(f"\n{'='*90}")
    print(f"{'GRAPH STRUCTURE SUMMARY':^90}")
    print(f"{'='*90}")
    print(f"{'Dataset':<10} {'Size':>12} {'Layers':>8} {'Total Edges':>14} {'L0 Nodes':>12} {'L0 Edges':>12}")
    print(f"{'-'*90}")

    for name, res in all_results.items():
        l0_stats = res.get('layer_graph_stats', {}).get(0, {})
        print(f"{name:<10} {res['dataset_size']:>12,} {res['num_layers']:>8} "
            f"{res.get('total_edges', 0):>14,} "
            f"{l0_stats.get('num_nodes', 0):>12,} "
            f"{l0_stats.get('num_edges', 0):>12,}")

    print(f"{'='*90}\n")
    
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        run_all_datasets()
    else:
        main()