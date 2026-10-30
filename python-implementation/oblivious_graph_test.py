from smart_pointer import SmartPointer
from single_access_machine import get_global_counter, print_stats_by_structure, print_max_write_batches, reset, reset_write_batches
from networkx import karate_club_graph, erdos_renyi_graph, DiGraph, Graph, connected_components, get_edge_attributes, write_edgelist
from typing import Dict, Set
from oblivious_graph import OGraph, Vertex, build_from_networkx, benchmark
import random

"""
Test algorithms for ensuring the OGraph algorithms correctly
Uses Karate Club graph: https://networkx.org/documentation/stable/auto_examples/graph/plot_karate_club.html 
"""


def test_emulate_graph_batches_multiwrite_destination_copies() -> None:
    """Static emulation makes one copy call per multi-write destination."""
    graph = DiGraph()
    graph.add_edges_from(
        [(0, 3), (1, 3), (2, 3), (0, 2), (1, 2)]
    )

    for pointer_type in ("multiwrite", "multiwriterary"):
        reset()
        SmartPointer.set_smart_pointer(pointer_type)
        if pointer_type == "multiwriterary":
            SmartPointer.set_branching_factor(4)

        ograph = OGraph(branching_factor=2, avl_tree_osam=False)
        v_ids: Dict[int, int] = {}
        entry_points: Dict[int, SmartPointer] = {}
        for name in graph.nodes:
            ograph.add_vertex(
                name, v_ids, entry_points, len(list(graph.neighbors(name)))
            )

        original_copy = SmartPointer.copy
        entry_point_ids = {
            id(pointer): vertex_id
            for vertex_id, pointer in entry_points.items()
        }
        batch_calls = []

        def tracking_copy(value, temp_copy=False, num_copies=1):
            if id(value) in entry_point_ids:
                batch_calls.append((entry_point_ids[id(value)], num_copies))
            return original_copy(value, temp_copy, num_copies)

        SmartPointer.copy = staticmethod(tracking_copy)
        try:
            ograph._OGraph__emulate_graph(graph, v_ids, entry_points)
        finally:
            SmartPointer.copy = staticmethod(original_copy)

        assert sorted(batch_calls) == sorted(
            [(v_ids[3], 3), (v_ids[2], 2)]
        )
        assert ograph.ogV_counts[v_ids[3]] == 4
        assert ograph.ogV_counts[v_ids[2]] == 3


def test_vertex_smart_copy_multiple() -> None:
    """Vertex copies batch each multi-write child pointer across all copies."""
    for pointer_type, expected_copy_reads in (
        ("multiwrite", 0),
        ("multiwriterary", 1),
    ):
        reset()
        SmartPointer.set_smart_pointer(pointer_type)
        if pointer_type == "multiwriterary":
            SmartPointer.set_branching_factor(4)

        child = SmartPointer(SmartPointer.new("child"))
        vertex = Vertex(7, out_children=[child, None], out_degree=1)
        before = get_global_counter()
        copies = SmartPointer.copy(vertex, num_copies=3)
        after = get_global_counter()

        assert isinstance(copies, list)
        assert len(copies) == 3
        assert after[1] - before[1] == expected_copy_reads
        assert all(copy.id == vertex.id for copy in copies)
        assert all(len(copy.out_children) == 2 for copy in copies)
        assert all(
            SmartPointer.get(copy.out_children[0]) == "child"
            for copy in copies
        )

def test_ograph(trials: int = 10, seed: int = 1) -> None:
    """Test building and traversing an Oblivious Graph"""
    # parameters to test on
    graph = karate_club_graph()
    name = "Karate Club"
    queue_osam = True
    stack_osam = True
    avl_tree_osam = True
    prime_graph = True
    branching_factor = 3
    pointers = {"original": [False, True], "multiwrite": [True], "recursive": [True]}
    algorithms = ["bfs", "dfs", "prim", "dijkstra", "random walk", "contact discovery", "directed triangle count", "pagerank"]
    make_copies_plural = [True, False]
    static_insertion_plural = [True, False]

    # iterate over every SmartPointer setting
    for pointer, move_semantics_plural in pointers.items():
        SmartPointer.set_smart_pointer(pointer)
        for move_semantics in move_semantics_plural:
            # toggle eager smart-copying
            for make_copies in make_copies_plural:
                # toggle static and dynamic insertion
                for static_insertion in static_insertion_plural:
                    # initialize ograph
                    ograph = build_from_networkx(    
                        function=karate_club_graph,
                        name=name, 
                        graph=None, 
                        args=None, 
                        branching_factor=branching_factor,
                        static_insertion=static_insertion, 
                        move_semantics=move_semantics, 
                        make_copies=make_copies,
                        use_clustering=False,
                        prime_graph=prime_graph,
                        seed=seed, 
                        connect_all=True, 
                        queue_osam=queue_osam,
                        stack_osam=stack_osam,
                        avl_tree_osam=avl_tree_osam,
                        random_weight_min=None, 
                        random_weight_max=None, 
                        sort_names=False, 
                        ordered_dynamic_vertices=True
                    )
                    
                    # call graph algorithm
                    for alg in algorithms:
                        traces = benchmark(  
                            ograph=ograph, 
                            name=name, 
                            trials=trials, 
                            algorithm=alg,
                            entry_point=0, 
                            next_entry_point=0, 
                            pick_random_entry_points=True,
                            move_semantics=move_semantics, 
                            make_copies=make_copies, 
                            walk_length=50,
                            damping_factor=0.9, 
                            queue_osam=queue_osam,
                            stack_osam=stack_osam,
                            avl_tree_osam=avl_tree_osam,
                            print_trace=True, 
                            print_each_trial=False
                        )
                        
                        # ensure the number of expected vertices appears
                        for i in range(34):
                            # print(i, ograph.ogV_counts[i], len(list(graph.neighbors(i))) + 1)
                            assert ograph.ogV_counts[i] == len(list(graph.neighbors(i))) + 1

def test_ograph_edge_deletion(fraction: float = 0.1, seed: int = 1) -> None:
    """Test deleting edges in an Oblivious Graph"""
    # parameters to test on
    graph = karate_club_graph()
    name = "Karate Club"
    queue_osam = True
    stack_osam = True
    avl_tree_osam = True
    prime_graph = True
    branching_factor = 3
    pointers = {"original": [False, True], "multiwrite": [True], "recursive": [True]}
    make_copies_plural = [True, False]
    static_insertion_plural = [True, False]

    # pick random edges to delete
    deleted_edges = []
    present_edges = list(graph.edges)
    num_edges_to_remove = int(len(present_edges)*fraction)
    for i in range(num_edges_to_remove):
        index = random.randint(0, len(present_edges)-1-i)
        deleted_edges.append(present_edges.pop(index))

    # create dictionary of deleted edges 
    deleted_edges_dict: Dict[int, Set[int]] = dict()
    for edge in deleted_edges:
        src_name, dst_name = edge
        if src_name in deleted_edges_dict:
            deleted_edges_dict[src_name].add(dst_name)
        else:
            deleted_edges_dict[src_name] = {dst_name}

        if dst_name in deleted_edges_dict:
            deleted_edges_dict[dst_name].add(src_name)
        else:
            deleted_edges_dict[dst_name] = {src_name}

    # iterate over every SmartPointer setting
    for pointer, move_semantics_plural in pointers.items():
        SmartPointer.set_smart_pointer(pointer)
        for move_semantics in move_semantics_plural:
            # toggle eager smart-copying
            for static_insertion in static_insertion_plural:
                # toggle static and dynamic insertion
                for make_copies in make_copies_plural:
                    # initialize ograph
                    ograph = build_from_networkx(    
                        function=karate_club_graph,
                        name=name, 
                        graph=None, 
                        args=None, 
                        branching_factor=branching_factor,
                        static_insertion=static_insertion, 
                        move_semantics=move_semantics, 
                        make_copies=make_copies,
                        use_clustering=False,
                        prime_graph=prime_graph,
                        seed=seed, 
                        connect_all=True, 
                        queue_osam=queue_osam,
                        stack_osam=stack_osam,
                        avl_tree_osam=avl_tree_osam,
                        random_weight_min=None, 
                        random_weight_max=None, 
                        sort_names=False, 
                        ordered_dynamic_vertices=True
                    )
                    
                    reset_write_batches()

                    # delete every edge selected for deletion
                    for edge in deleted_edges:
                        src_name, dst_name = edge
                        print(f"Deleting edge {src_name}->{dst_name}")
                        print(f"Deleting edge {dst_name}->{src_name}")
                        if move_semantics:
                            ograph.delete_edge(src_name, dst_name)     
                            ograph.delete_edge(dst_name, src_name)        
                        else:
                            ograph.delete_edge_full_copy(src_name, dst_name)     
                            ograph.delete_edge_full_copy(dst_name, src_name)    
                    
                    print(f"\nPost delete edge stats {get_global_counter()}", flush=True)
                    print_stats_by_structure(flush=True)
                    print_max_write_batches(flush=True)
                    reset_write_batches()

                    # call add_neighbors to search OutEdge tree and ensure deletions occurred
                    for src_name in deleted_edges_dict:
                        dst_names = deleted_edges_dict[src_name]
                        present_neighbors: Set[int] = set()
                        p_src = ograph.entry_points[ograph.v_ids[src_name]]
                        if move_semantics:
                            ograph.add_neighbors(p_src, "SET", present_neighbors, "DST_ID", None, make_copies)
                        else:
                            ogV = SmartPointer.get_and_copy(p_src)
                            ograph.add_neighbors_full_copy(p_src, ogV, "SET", present_neighbors, "DST_ID", None)
                        assert not present_neighbors & dst_names

                        print(f"Neighbors of vertex {src_name}: {present_neighbors}")

                    print(f"\nPost OutEdge tree search stats {get_global_counter()}", flush=True)
                    print_stats_by_structure(flush=True)
                    print_max_write_batches(flush=True)

                    # ensure the number of expected vertices appears
                    for i in range(34):
                        if i in deleted_edges_dict:
                            # print(i, ograph.ogV_counts[i], len(list(graph.neighbors(i))) + 1 - len(deleted_edges_dict[i]))
                            assert ograph.ogV_counts[i] == len(list(graph.neighbors(i))) + 1 - len(deleted_edges_dict[i])
                        else:
                            # print(i, ograph.ogV_counts[i], len(list(graph.neighbors(i))) + 1)
                            assert ograph.ogV_counts[i] == len(list(graph.neighbors(i))) + 1

def test_ograph_vertex_deletion(fraction: float = 0.5, seed: int = 1) -> None:
    """Test deleting vertices in an Oblivious Graph"""
    # parameters to test on
    graph = karate_club_graph()
    name = "Karate Club"
    queue_osam = True
    stack_osam = True
    avl_tree_osam = True
    prime_graph = True
    branching_factor = 3
    pointers = {"original": [False, True], "multiwrite": [True], "recursive": [True]}
    make_copies_plural = [True, False]
    static_insertion_plural = [True, False]

    # pick random vertices to delete
    deleted_vertices = set()
    present_vertices = list(graph.nodes)
    num_vertices_to_remove = int(len(present_vertices)*fraction)
    for i in range(num_vertices_to_remove):
        index = random.randint(0, len(present_vertices)-1-i)
        deleted_vertices.add(present_vertices.pop(index))
    
    # iterate over every SmartPointer setting
    for pointer, move_semantics_plural in pointers.items():
        SmartPointer.set_smart_pointer(pointer)
        for move_semantics in move_semantics_plural:
            # toggle eager smart-copying
            for make_copies in [False]:
                # toggle static and dynamic insertion
                for static_insertion in [False]:
                    # initialize ograph
                    ograph = build_from_networkx(    
                        function=karate_club_graph,
                        name=name, 
                        graph=None, 
                        args=None, 
                        branching_factor=branching_factor,
                        static_insertion=static_insertion, 
                        move_semantics=move_semantics, 
                        make_copies=make_copies,
                        use_clustering=False,
                        prime_graph=prime_graph,
                        seed=seed, 
                        connect_all=True, 
                        queue_osam=queue_osam,
                        stack_osam=stack_osam,
                        avl_tree_osam=avl_tree_osam,
                        random_weight_min=None, 
                        random_weight_max=None, 
                        sort_names=False, 
                        ordered_dynamic_vertices=True
                    )

                    reset_write_batches()

                    # delete vertex
                    for src_name in deleted_vertices:
                        print(f"Deleting vertex {src_name}")
                        if move_semantics:
                            ograph.delete_vertex(src_name)
                        else:
                            ograph.delete_vertex_full_copy(src_name)

                    print(f"\nNumber of deleted objects: {ograph.num_deleted_objs}")

                    print(f"\nPost delete vertex stats {get_global_counter()}", flush=True)
                    print_stats_by_structure(flush=True)
                    print_max_write_batches(flush=True)
                    reset_write_batches()

                    # check OutEdge tree to ensure all deleted verices are gone
                    remaining_neighbors = dict()
                    for src_name in present_vertices:
                        present_neighbors: Set[int] = set()
                        p_src = ograph.entry_points[ograph.v_ids[src_name]]
                        if move_semantics:
                            ograph.add_neighbors(p_src, "SET", present_neighbors, "DST_ID", None, make_copies)
                        else:
                            ogV = SmartPointer.get_and_copy(p_src)
                            ograph.add_neighbors_full_copy(p_src, ogV, "SET", present_neighbors, "DST_ID", None)
                        assert not present_neighbors & deleted_vertices
                        remaining_neighbors[src_name] = present_neighbors
                        print(f"Neighbors of vertex {src_name}: {present_neighbors}")

                    print(f"\nPost OutEdge tree search stats {get_global_counter()}", flush=True)
                    print_stats_by_structure(flush=True)
                    print_max_write_batches(flush=True)
                        
                    # all references to deleted objects should have been encountered and removed
                    assert ograph.num_deleted_objs == 0

                    # ensure the number of expected vertices appears
                    for i in range(34):
                        if i in deleted_vertices:
                            assert ograph.ogV_counts[i] == 0
                        else:
                            assert ograph.ogV_counts[i] == len((set(graph.neighbors(i)) & remaining_neighbors[i])) + 1
                            
def test_ograph_clustered(trials: int = 10, seed: int = 1) -> None:
    """Test Clusters on an Oblivious Graph"""
    graph = karate_club_graph()
    name = "Karate Club"
    queue_osam = True
    stack_osam = True
    avl_tree_osam = True
    prime_graph = False
    use_clustering = True
    branching_factor = 3
    pointers = ["original", "multiwrite", "recursive"]
    algorithms = ["bfs", "dfs", "prim", "dijkstra", "random walk", "contact discovery", "directed triangle count", "pagerank"]
    make_copies_plural = [True, False]
    move_semantics = True
    static_insertion = True

    for pointer in pointers:
        SmartPointer.set_smart_pointer(pointer)
        for make_copies in make_copies_plural:
            # initialize ograph
            ograph = build_from_networkx(    
                function=karate_club_graph,
                name=name, 
                graph=None, 
                args=None, 
                branching_factor=branching_factor,
                static_insertion=static_insertion, 
                move_semantics=move_semantics, 
                make_copies=make_copies,
                use_clustering=use_clustering,
                prime_graph=prime_graph,
                seed=seed, 
                connect_all=True, 
                queue_osam=queue_osam,
                stack_osam=stack_osam,
                avl_tree_osam=avl_tree_osam,
                random_weight_min=None, 
                random_weight_max=None, 
                sort_names=False, 
                ordered_dynamic_vertices=True
            )

            # collect cluster counts before any algorithm is run
            cluster_counts = dict()
            for community, count in ograph.cluster_counts.items():
                cluster_counts[community] = count

            # call graph algorithm
            for alg in algorithms:
                traces = benchmark(  
                    ograph=ograph, 
                    name=name, 
                    trials=trials, 
                    algorithm=alg,
                    entry_point=0, 
                    next_entry_point=0, 
                    pick_random_entry_points=True,
                    move_semantics=move_semantics, 
                    make_copies=make_copies, 
                    walk_length=50,
                    damping_factor=0.9, 
                    queue_osam=queue_osam,
                    stack_osam=stack_osam,
                    avl_tree_osam=avl_tree_osam,
                    print_trace=True, 
                    print_each_trial=False
                )

                # ensure the number of expected vertices appears
                for i in range(34):
                    # print(i, ograph.ogV_counts[i], len(list(graph.neighbors(i))) + 1)
                    assert ograph.ogV_counts[i] == len(list(graph.neighbors(i))) + 1

                for community, count in cluster_counts.items():
                    # print(community, ograph.cluster_counts[community], count)
                    assert ograph.cluster_counts[community] == count


if __name__ == "__main__":
    seed = 1
    trials = 10
    test_emulate_graph_batches_multiwrite_destination_copies()
    test_vertex_smart_copy_multiple()
    test_ograph(trials, seed=seed)
    test_ograph_edge_deletion(seed=seed)
    test_ograph_vertex_deletion(seed=seed)
    test_ograph_clustered(trials, seed=seed)
