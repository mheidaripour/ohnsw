import re
import argparse
import os
from math import ceil, log
from detect_failed_tests import detect_failed_tests
from typing import List, Dict, Tuple, Any
from sys import maxsize

"""
The recursive calculation is as follows:

total_allocs = allocs of build and every other phase - SmartAVLTree allocs
recursion_base = bs / 8
levels = 1 if bs == 4096 else 4
recursion_levels = ceil( max( log(total_allocs, recursion_base) - levels, 1 ) )
raw_phase_reads = raw_phase_reads - build_reads
phase_avl_tree_reads = phase_avl_tree_reads - build_avl_tree_reads
recursive_phase_reads = (raw_phase_reads - phase_avl_tree_reads) * recursion_levels + phase_avl_tree_reads
final_data_point = log( recursive_phase_reads/50, 2)
"""

official_pointer_names = {
    "originalfalse": "Original",
    "originaltrue": "Original w/ Move",
    "multiwritetrue": "Multi-Write",
    "recursivetrue": "Recursive",
}

def read_all_logs(
        logs_directory: str
    ) -> Dict[str, Dict[str, Dict[str, List[int] | Dict[str, List[int]]]]]:
    """Parse and collect all stats for every benchmark"""
    # get all log files
    logs = sorted(os.listdir(logs_directory))

    failed_logs = detect_failed_tests(logs_directory)
    for log in failed_logs:
        logs.remove(log)

    # all_plot_data: a multi-layered dictionary that stores SAM accesses for all logs
    # all_plot_data mapping guide:
    # key1: plot_parameters -> the parameters used to build a single plot such as
    #                          d, bs, trials, make_copies, prime_graph, and static_insertion
    #   value1: {key2: value2}
    #   key2: data_point_parameters -> the variables that represent a single data point 
    #                                  such as n, pointer, move_sematics, and algorithm
    #                                  and optionally walk_length and damping factor
    #       value2: {key3: value3}
    #       key3: 
    #           case 1: non-build algorithm -> stores overall SAM stats incurred during a phase
    #               value3: [allocs, reads, writes]
    #           case 2: algorithm by structure -> provides a breakdown of stats incurred during
    #                                           a phase by a specific SAM-based structure
    #               value3: {structure: [allocs, reads, writes]}
    #           case 3: build -> collects all build stats and averages them out
    #                            while build stats should be the same per algorithm
    #                            when using a fixed seed, this still accounts for fluctuations
    #               value3: [allocs, reads, writes, # of builds phases to divide by]
    #           case 4: build by structure -> collects all build stats for each structure
    #                                         and computes averages
    #               value3: {structure: [allocs, reads, writes]}
    all_plot_data: Dict[str, Dict[str, Dict[str, List[int] | Dict[str, List[int]]]]] = dict()

    for log in logs:
        name = log.split(".log")[0]
        log_parameters = name.split("_")

        # parameters that will be fixed for a single plot
        d_tag = log_parameters[2]
        bs_tag = log_parameters[3]
        trials_tag = log_parameters[4]
        make_copies_tag = log_parameters[8]
        prime_graph_tag = log_parameters[9]
        static_insertion_tag = log_parameters[10]

        # tagline to represent a single graph to be made 
        plot_parameters = f"{d_tag}_{bs_tag}_{trials_tag}_{make_copies_tag}_{prime_graph_tag}_{static_insertion_tag}"

        # dynamic parameters that place a single point on a plot
        n_tag = log_parameters[1]
        pointer_tag = log_parameters[5]
        algorithm_tag = log_parameters[6]
        move_semantics_tag = log_parameters[7]
        data_point_parameters = f"{n_tag}_{pointer_tag}_{move_semantics_tag}_{algorithm_tag}"

        # add walk length and/or damping factor if relevant
        if len(log_parameters) == 13:
            data_point_parameters += f"_{log_parameters[11]}_{log_parameters[12]}"
        elif len(log_parameters) == 12:
            data_point_parameters += f"_{log_parameters[11]}"

        # grab algorithm name without "alg-" prefix
        algorithm = algorithm_tag.split("-")[1]

        # collect SAM access stats from a single file
        result = read_single_log(logs_directory, log, algorithm, data_point_parameters)
        if result is None: 
            continue
        single_plot_data, build_stats_list, build_stats_by_structure = result

        build_data_point_parameters = f"{n_tag}_{pointer_tag}_{move_semantics_tag}_alg-build"  

        # add new data point to be plotted later
        if plot_parameters in all_plot_data:
            # this set of plot parameters already exists
            all_plot_data[plot_parameters][data_point_parameters] = single_plot_data
            
            if build_data_point_parameters in all_plot_data[plot_parameters]:
                # collect all build stats and average out later
                build_stats_dict = all_plot_data[plot_parameters][build_data_point_parameters]

                # sum build stats
                assert isinstance(build_stats_dict["build"], list)
                for i in range(4):
                    build_stats_dict["build"][i] += build_stats_list[i]

                # sum build stats per structure
                build_by_structure_dict = build_stats_dict["build by structure"]
                assert isinstance(build_by_structure_dict, dict)
                for structure in build_by_structure_dict:
                    structure_stats_list = build_by_structure_dict[structure]
                    assert isinstance(structure_stats_list, list)
                    for i in range(3):
                        structure_stats_list[i] += build_stats_by_structure[structure][i]
            
            else:
                # insert new build stats
                all_plot_data[plot_parameters][build_data_point_parameters] = {
                    "build": build_stats_list,
                    "build by structure": build_stats_by_structure
                }

        else:
            # initialize new graph entry
            all_plot_data[plot_parameters] = { 
                data_point_parameters: single_plot_data,
                build_data_point_parameters: {
                    "build": build_stats_list,
                    "build by structure": build_stats_by_structure
                }
            }

    return all_plot_data

def read_single_log(
        logs_directory: str, 
        log: str, 
        algorithm: str, 
        data_point_parameters: str
    ) -> None | Tuple[Dict[str, List[int] | Dict[str, List[int]]], List[int], Dict[str, List[int]]]:
    """Parse the inputs of a single log file"""
    # get content of log file
    f = open(f"{logs_directory}/{log}")
    log_text = f.read()
    f.close()

    # stat collection storage
    build_stats_list: List[int]
    alg_stats: List[int]
    structure_stats: List[int]
    build_stats_by_structure: Dict[str, List[int]] = dict()
    alg_stats_by_structure: Dict[str, List[int]] = dict()

    # iterate over each line in the log and collect SAM access stats
    i = 0
    lines = log_text.strip().splitlines()

    while i < len(lines):
        line = lines[i]
        i += 1

        if "Skipping test: p=" in line:
            return None

        elif line.startswith("Post build stats"):
            build_stats_list = list(map(int, re.findall(r"\d+", line)))
            # add reference counter to track number of builds phases to divide by
            build_stats_list.append(1) 

        elif line.startswith("Post build structure stats:"):
            # track stats per structure that incurs SAM operations
            line = lines[i]
            while "Structure stats:" in line:
                structure = line[17:].split(" ")[0]
                structure_stats = list(map(int, re.findall(r"\d+", line)))
                build_stats_by_structure[structure] = structure_stats
                i += 1
                line = lines[i]

        elif line.startswith("Post algorithm"):
            alg_stats = list(map(int, re.findall(r"\d+", line)))
        
        elif line.startswith("Post") and line.endswith("structure stats:"):
            # track stats per object that incurs SAM operations
            line = lines[i]
            while "Structure stats:" in line:
                structure = line[17:].split(" ")[0]
                structure_stats = list(map(int, re.findall(r"\d+", line)))
                alg_stats_by_structure[structure] = structure_stats
                i += 1
                line = lines[i]

    # the algorithm stats contain stats incurred from building
    # subtract build stats to see the algorithm stats
    for i in range(3):
        alg_stats[i] -= build_stats_list[i]

    for structure in alg_stats_by_structure.keys():
        for i in range(3):
            try:
                alg_stats_by_structure[structure][i] -= build_stats_by_structure[structure][i]
            except:
                pass
    
    single_plot_data: Dict[str, List[int] | Dict[str, List[int]]] = {
        algorithm: alg_stats,
        f"{algorithm} by structure": alg_stats_by_structure
    }
    return single_plot_data, build_stats_list, build_stats_by_structure

def average_and_scale_stats(
        all_plot_data: Dict[str, Dict[str, Dict[str, List[int] | Dict[str, List[int]]]]]
    ) -> None:
    """
    - Averages the stats for builds across the number of build phases encountered
    Every benchmark involves building the graph and running one algorithm
    By default, a fixed seed means the stats should be the same for all benchmarks

    - Scales the reads for recursive benchmark by some factor to simulate
    the overhead a standard PathORAM would incur
    """
    for plot_parameters in all_plot_data:
        plot_parameters_dict = all_plot_data[plot_parameters]

        # average the number of accesses per build
        # these could be different for each algorithm if
        # the seed value was adjusted 
        for data_point_parameters in plot_parameters_dict:
            if data_point_parameters.endswith("_alg-build"):
                # average all build stats
                build_stats_dict = plot_parameters_dict[data_point_parameters]
                assert isinstance(build_stats_dict["build"], list)
                number_of_builds = build_stats_dict["build"][3]
                for i in range(3):
                    build_stats_dict["build"][i] = int(build_stats_dict["build"][i] / number_of_builds)

                # average all build stats per structure user
                build_by_structure_dict = build_stats_dict["build by structure"]
                assert isinstance(build_by_structure_dict, dict)
                for structure in build_by_structure_dict:
                    for i in range(3):
                        build_by_structure_dict[structure][i] = int(build_by_structure_dict[structure][i] / number_of_builds ) 
            
        # collect recursive tests names so that we can scale them
        # maps data point parameters to lists of algorithms 
        recursive_data_points_to_alg: Dict[str, List[str]] = dict() 
        for data_point_parameters in plot_parameters_dict:
            if "recursive" not in data_point_parameters:
                continue

            data_point_parameters_split = data_point_parameters.split("_")
            data_point_no_alg = f"{data_point_parameters_split[0]}_{data_point_parameters_split[1]}_{data_point_parameters_split[2]}"
            
            algorithm_tag = data_point_parameters_split[3]
            for i in range(4, len(data_point_parameters_split)):
                algorithm_tag += f"_{data_point_parameters_split[i]}"

            if data_point_no_alg in recursive_data_points_to_alg:
                recursive_data_points_to_alg[data_point_no_alg].append(algorithm_tag)

            else:
                recursive_data_points_to_alg[data_point_no_alg] = [algorithm_tag]
        
        # scale the reads of the recursive implementation to match 
        # the performance expected from a stock PathORAM scheme
        bs = int(plot_parameters.split("_")[1].split("-")[1])
        for data_point_no_alg in recursive_data_points_to_alg:
            total_allocs = 0

            # sum up total allocs 
            # subtract all allocs for SmartAVLTrees
            for algorithm_tag in recursive_data_points_to_alg[data_point_no_alg]:
                data_point_parameters = f"{data_point_no_alg}_{algorithm_tag}"
                alg_stats_dict = plot_parameters_dict[data_point_parameters]
            
                # get algorithm and build strings to look up other useful stats
                algorithm = algorithm_tag.split("_")[0].split("-")[1]

                # add allocs from graph phase
                alg_stats = alg_stats_dict[algorithm]
                assert isinstance(alg_stats, list)
                total_allocs += alg_stats[0]
                
                # exclude SmartAVLTree accesses from the recursive scaling
                try:
                    alg_stats_by_structure = alg_stats_dict[f"{algorithm} by structure"]
                    assert isinstance(alg_stats_by_structure, dict)
                    avl_tree_alg_allocs = alg_stats_by_structure["SmartAVLTree"][0]
                except:
                    avl_tree_alg_allocs = 0
                total_allocs -= avl_tree_alg_allocs

            # calculate recursive base 
            recursion_base = bs / 8
            recursion_levels = log(total_allocs, recursion_base)
            if bs == 4096:
                recursion_levels = ceil(max(recursion_levels-1, 1))
            else:
                recursion_levels = ceil(max(recursion_levels-4, 1))

            # scale each function by recursion factor
            for algorithm_tag in recursive_data_points_to_alg[data_point_no_alg]:
                data_point_parameters = f"{data_point_no_alg}_{algorithm_tag}"
                alg_stats_dict = plot_parameters_dict[data_point_parameters]

                # get algorithm and build strings to look up other useful stats
                algorithm = algorithm_tag.split("_")[0].split("-")[1]

                # exclude SmartAVLTree accesses
                try:
                    alg_stats_by_structure = alg_stats_dict[f"{algorithm} by structure"]
                    assert isinstance(alg_stats_by_structure, dict)
                    avl_tree_alg_reads = alg_stats_by_structure["SmartAVLTree"][1]
                except:
                    avl_tree_alg_reads = 0

                # scale recursive reads for each function
                alg_stats = alg_stats_dict[algorithm]
                assert isinstance(alg_stats, list)
                alg_stats[1] = (alg_stats[1] - avl_tree_alg_reads) * recursion_levels + avl_tree_alg_reads

        # average algorithm stats across each trial
        # ignores builds, which do not correspond to # of trials
        trials = int(plot_parameters.split("_")[2].split("-")[1])
        for data_point_parameters in plot_parameters_dict:
            if "build" in data_point_parameters:
                continue

            algorithm = data_point_parameters.split("_")[3].split("-")[1]
            alg_stats = plot_parameters_dict[data_point_parameters][algorithm]
            assert isinstance(alg_stats, list)
            for i in range(3):
                alg_stats[i] = int(alg_stats[i] / trials)
            
def generate_all_plots_and_tables(
        plots_directory: str, 
        tables_directory: str,
        stats_to_plot: List[str],
        all_plot_data: Dict[str, Dict[str, Dict[str, List[int] | Dict[str, List[int]]]]]
    ) -> None:
    """Produce every possible plot and table per log"""
    # create directories to store plots and tables if they do not exist
    if not os.path.exists(plots_directory):
        os.mkdir(plots_directory)
    if not os.path.exists(tables_directory):
        os.mkdir(tables_directory)

    for plot_parameters in all_plot_data:
        plot_parameters_dict = all_plot_data[plot_parameters]

        # remove tag from string so we only get the value of the attribute
        plot_parameters_list_no_tag = plot_parameters.split("_")
        for i in range(len(plot_parameters_list_no_tag)):
            plot_parameters_list_no_tag[i] = plot_parameters_list_no_tag[i].split("-")[1]

        # map each algorithm to all its data points so
        # we can plot data points by algorithm
        algs_to_data_points_dict: Dict[str, List[str]] = dict() 
        for data_point_parameters in plot_parameters_dict:
            data_point_parameters_split = data_point_parameters.split("_")
            data_point_no_alg = f"{data_point_parameters_split[0]}_{data_point_parameters_split[1]}_{data_point_parameters_split[2]}"
            
            # get algorithm name
            # include walk_length and damping_factor if relevant
            algorithm_tag = data_point_parameters_split[3]
            for i in range(4, len(data_point_parameters_split)):
                algorithm_tag += f"_{data_point_parameters_split[i]}"

            if algorithm_tag in algs_to_data_points_dict:
                algs_to_data_points_dict[algorithm_tag].append(data_point_no_alg)

            else:
                algs_to_data_points_dict[algorithm_tag] = [data_point_no_alg]

        # write plot outputs
        for algorithm_tag, data_point_no_alg_list in algs_to_data_points_dict.items():
            # create dictionary of pointers + move_semantics to n
            # so we can process each pointer per each graph size
            pointer_move_semantics_to_n: Dict[str, List[str]] = dict()
            for data_point_no_alg in data_point_no_alg_list:
                data_point_no_alg_split = data_point_no_alg.split("_")
                pointer_move_semantics = f"{data_point_no_alg_split[1]}_{data_point_no_alg_split[2]}"
                n = data_point_no_alg_split[0]
                
                # create list of n's
                if pointer_move_semantics in pointer_move_semantics_to_n:
                    pointer_move_semantics_to_n[pointer_move_semantics].append(n)
                else:
                    pointer_move_semantics_to_n[pointer_move_semantics] = [n]
            
            write_single_plots(plots_directory, stats_to_plot, algorithm_tag, pointer_move_semantics_to_n, plot_parameters_list_no_tag, plot_parameters_dict)
            write_single_tables(tables_directory, stats_to_plot, algorithm_tag, pointer_move_semantics_to_n, plot_parameters_list_no_tag, plot_parameters_dict)

def write_single_plots(
        plots_directory: str,
        stats_to_plot: List[str], 
        algorithm_tag: str, 
        pointer_move_semantics_to_n: Dict[str, List[str]], 
        plot_parameters_list_no_tag: List[str], 
        plot_parameters_dict: Dict[str, Dict[str, List[int] | Dict[str, List[int]]]]
    ) -> None:
    """Creates a single pgfplot that can be viewed in LaTeX"""
    # unpack parameters without file tags
    d, bs, trials, make_copies, prime_graph, static_insertion = plot_parameters_list_no_tag
    algorithm_tag_split = algorithm_tag.split("_")
    algorithm = algorithm_tag_split[0].split("-")[1]
    walk_length = damping_factor = None
    if len(algorithm_tag_split) >= 2:
        walk_length = algorithm_tag_split[1].split("-")[1]
    if len(algorithm_tag_split) >= 3:
        damping_factor = algorithm_tag_split[2].split("-")[1]

    # generate plots for all desired stats
    for stat in stats_to_plot:
        match stat:
            case "alloc":
                stat_index = 0
            case "read":
                stat_index = 1
            case "write":
                stat_index = 2
            case _:
                raise RuntimeError(f"'{stat}' is not a valid SAM stat!")

        # create new plot
        plot = f"log2n_vs_{algorithm}_{stat}s_d{d}_bs{bs}"
        if algorithm != "build":
            plot += f"_trials{trials}"
        plot += f"_copies{make_copies}_prime{prime_graph}_staticinsertion{static_insertion}"
        if isinstance(walk_length, str):
            plot += f"_wl{walk_length}"
        if isinstance(damping_factor, str):
            plot += f"_df{damping_factor}"
        plot += ".tex"

        # assemble plot content to be written to a tex file
        plot_content = "\\begin{tikzpicture}\n"
        plot_content += "\\begin{axis}[\n"
        plot_content += "xlabel={$\\log_2 n$}, ylabel={$\\log_2 (\\text{{" + f"{stat}" +"s}})$},\n"
        plot_content += "width=10cm, height=8cm,\n"
        plot_content += "grid=major,\n"

        # create legend based on which pointers are present
        sorted_pointer_move_semantics = sorted(list(pointer_move_semantics_to_n.keys()))
        legend = []
        for pointer_move_semantics in sorted_pointer_move_semantics:
            pointer_move_semantics_split = pointer_move_semantics.split("_")
            pointer = pointer_move_semantics_split[0].split("-")[1]
            move_semantics = pointer_move_semantics_split[1].split("-")[1]
            legend.append(official_pointer_names[f"{pointer}{move_semantics}"])
        legend.sort()
        plot_content += "legend entries={"+", ".join(legend)+"},\n"
        plot_content += "legend cell align={left},\n"
        plot_content += "legend style={at={(1.03,0.5)}, anchor=west},\n]\n"

        # write graph data points for each pointer type
        for pointer_move_semantics in sorted_pointer_move_semantics:
            plot_content += "\\addplot+[only marks] coordinates {\n"
            raw_data_points = []
            
            # calculate log of stats and square root of n 
            for n in pointer_move_semantics_to_n[pointer_move_semantics]:
                log2n = log(int(n.split("-")[1]), 2)
                data_point_no_alg_split = f"{n}_{pointer_move_semantics}_{algorithm_tag}"
                alg_stats = plot_parameters_dict[data_point_no_alg_split][algorithm]
                assert isinstance(alg_stats, list)
                stats: float | int = alg_stats[stat_index]
                stats = round(log(stats, 2), 2)
                raw_data_points.append((log2n, stats))

            # write stats in ascending order of n
            raw_data_points.sort(key=lambda map_item: map_item[0])
            for log2n, stats in raw_data_points:
                plot_content += f"({log2n}, {stats})\n"
            plot_content += "};\n"
        
        plot_content += "\\end{axis}\n"
        plot_content += "\\end{tikzpicture}\n"

        # write plot to tex file
        f = open(f"{plots_directory}/{plot}", "w+")
        f.write(plot_content)
        f.close()

def write_single_tables(
        tables_directory: str,
        stats_to_plot: List[str], 
        algorithm_tag: str, 
        pointer_move_semantics_to_n: Dict[str, List[str]],
        plot_parameters_list_no_tag: List[str], 
        plot_parameters_dict: Dict[str, Dict[str, List[int] | Dict[str, List[int]]]]
    ) -> None:
    """Create a table that displays all data points visible in one graph"""
    # unpack parameters without file tags
    d, bs, trials, make_copies, prime_graph, static_insertion = plot_parameters_list_no_tag
    algorithm_tag_split = algorithm_tag.split("_")
    algorithm = algorithm_tag_split[0].split("-")[1]
    walk_length = damping_factor = None
    if len(algorithm_tag_split) >= 2:
        walk_length = algorithm_tag_split[1].split("-")[1]
    if len(algorithm_tag_split) >= 3:
        damping_factor = algorithm_tag_split[2].split("-")[1]

    # generate tables for all desired statss
    for stat in stats_to_plot:
        match stat:
            case "alloc":
                stat_index = 0
            case "read":
                stat_index = 1
            case "write":
                stat_index = 2
            case _:
                raise RuntimeError(f"'{stat}' is not a valid SAM stat!")

        # create new table
        table = f"{stat}s_by_structure_{algorithm}_d{d}_bs{bs}"
        if algorithm != "build":
            table += f"_trials{trials}"
        table += f"_copies{make_copies}_prime{prime_graph}_staticinsertion{static_insertion}"
        if isinstance(walk_length, str):
            table += f"_wl{walk_length}"
        if isinstance(damping_factor, str):
            table += f"_df{damping_factor}"
        table += ".tex"

        # build several tables to write to one file
        table_content = ""

        # build separate tables for each pointer type
        for pointer_move_semantics, n_tags in pointer_move_semantics_to_n.items():
            # collect all structures that contributed reads during a phase
            # exclude structures that did not incur anything 
            # additionally collect smallest and largest ns so 
            # we know how many indices are needed for the table
            structures = set()
            min_n = maxsize
            max_n = -1
            for n_tag in n_tags:
                data_point_parameters = f"{n_tag}_{pointer_move_semantics}_{algorithm_tag}"
                structure_stats_dict = plot_parameters_dict[data_point_parameters][f"{algorithm} by structure"]
                assert isinstance(structure_stats_dict, dict)
                for structure in structure_stats_dict:
                    structure_stats_list = structure_stats_dict[structure]
                    assert isinstance(structure_stats_list, list)
                    if structure_stats_list[stat_index] > 0:
                        structures.add(structure)

                n = int(n_tag.split("-")[1])
                min_n = min(min_n, n)
                max_n = max(max_n, n)

            # write table up to first containing all structures
            sorted_structures = sorted(list(structures))
            table_content += "\\begin{table}[]\n"
            table_content += "\\centering\n"
            table_content += "\\begin{tabular}{|l|" + "c|"*len(sorted_structures) + "}\n$log_2(n)$ "
            for structure in sorted_structures:
                table_content += f"& {structure} "
            table_content += "\\\\ \\hline\n"

            # enumerate stats for each structure as n increases
            min_power = int(log(min_n, 2))
            max_power = int(log(max_n, 2))
            for power in range(min_power, max_power+1):
                table_content += f"{power} "
                data_point_parameters = f"n-{pow(2, power)}_{pointer_move_semantics}_{algorithm_tag}"
                for structure in sorted_structures:
                    try:
                        alg_stats_by_structure = plot_parameters_dict[data_point_parameters][f"{algorithm} by structure"]
                        assert isinstance(alg_stats_by_structure, dict)
                        stats = alg_stats_by_structure[structure][stat_index]
                    except:
                        stats = 0
                    table_content += f"& {stats} "
                table_content += "\\\\ \\hline\n"

            # build accurate caption                    
            table_content += "\\end{tabular}\n"
            table_content += "\\caption{Breakdown of " + f"{stat}s by structure for "
            match algorithm:
                case "build":
                    table_content += "Building"
                case "bfs":
                    table_content += "BFS"
                case "dfs":
                    table_content += "DFS"
                case "dijkstra":
                    table_content += "Dijkstra"
                case "prim":
                    table_content += "Prim"
                case "rw":
                    table_content += "Random Walk"
                case "cd":
                    table_content += "Contact Discovery"
                case "dtc":
                    table_content += "Directed Triangle Count"
                case "pr":
                    table_content += "PageRank"

            pointer_move_semantics_split = pointer_move_semantics.split("_")
            pointer = pointer_move_semantics_split[0].split("-")[1]
            move_semantics = pointer_move_semantics_split[1].split("-")[1]
            table_content += f" on {official_pointer_names[f"{pointer}{move_semantics}"]} architecture"
            if algorithm != "build":
                table_content += f" across {trials} trials"
            table_content += f" with degree $d={d}$, block size $bs={bs}$,"

            if isinstance(walk_length, str):
                table_content += f" walk length $wl={walk_length}$,"
            if isinstance(damping_factor, str):
                table_content += f" damping factor $df={damping_factor}$,"
            
            make_copies_result = "enabled" if make_copies == "true" else "disabled"
            prime_graph_result = "enabled" if prime_graph == "true" else "disabled"
            static_insertion_result = "enabled" if static_insertion == "true" else "disabled"
            table_content += f" smart copying {make_copies_result}, priming {prime_graph_result}, and static building {static_insertion_result}"
            table_content += ".}\n"
            table_content += "    \\label{tab:"
            table_content += f"{table.split(".tex")[0]}_pt{pointer}_move{move_semantics}"
            table_content += "}\n\\end{table}\n\n"

        # write table
        f = open(f"{tables_directory}/{table}", "w+")
        f.write(table_content)
        f.close()

def write_main_plot(
        plots_directory: str, 
        plot_path: str,
        main_plot_name: str
    ) -> None:
    """Writes a tex file that contains every generated pgf plot as a figure"""
    plots = sorted(list(os.listdir(plots_directory)))

    # add each plot to the tex file
    main_plot = ""
    for plot in plots:
        plot_no_extension = plot.split(".tex")[0]
        main_plot += "\n\\begin{figure}[ht]\n"
        main_plot += "    \\centering\n"
        main_plot += "    \\resizebox{.9\\columnwidth}{!}{\\input{"
        main_plot += f"{plot_path}/{plot_no_extension}" + "}}\n"

        # parse parameters from file name
        plot_split = plot_no_extension.split("_")
        algorithm = plot_split[2]
        d = plot_split[4][1:]
        bs = plot_split[5][2:]

        # trials is not specified during the build phase
        i = 6
        if algorithm != "build":
            trials = plot_split[i][6:]
            i += 1
        else:
            trials = None

        make_copies = plot_split[i][6:]
        prime_graph = plot_split[i+1][5:]
        static_insertion = plot_split[i+2][15:]
        walk_length = damping_factor = None
        if len(plot_split) >= i+4:
            walk_length = plot_split[i+3][2:]
        if len(plot_split) >= i+5:
            damping_factor = plot_split[i+4][2:]

        # generate accurate caption
        main_plot += "    \\caption{"
        match algorithm:
            case "build":
                main_plot += "Building"
            case "bfs":
                main_plot += "BFS"
            case "dfs":
                main_plot += "DFS"
            case "dijkstra":
                main_plot += "Dijkstra"
            case "prim":
                main_plot += "Prim"
            case "rw":
                main_plot += "Random Walk"
            case "cd":
                main_plot += "Contact Discovery"
            case "dtc":
                main_plot += "Directed Triangle Count"
            case "pr":
                main_plot += "PageRank"

        if isinstance(trials, str):
            main_plot += f" averaged across {trials} trials"

        main_plot += f" with degree $d={d}$, block size $bs={bs}$,"

        if isinstance(walk_length, str):
            main_plot += f" walk length $wl={walk_length}$,"
        if isinstance(damping_factor, str):
            main_plot += f" damping factor $df={damping_factor}$,"

        make_copies_result = "enabled" if make_copies == "true" else "disabled"
        prime_graph_result = "enabled" if prime_graph == "true" else "disabled"
        static_insertion_result = "enabled" if static_insertion == "true" else "disabled"
        main_plot += f" smart copying {make_copies_result}, priming {prime_graph_result}, and static building {static_insertion_result}"
        main_plot += ".}\n"
        main_plot += "    \\label{fig:"
        main_plot += f"{plot.split(".tex")[0]}"
        main_plot += "}\n\\end{figure}\n"

    # output main plot
    f = open(f"{main_plot_name}", "w+")
    f.write(main_plot)
    f.close()

def write_main_table(
        tables_directory: str, 
        table_path: str,
        main_table_name: str
    ) -> None:
    """Writes a tex file that includes every generated table"""
    tables = sorted(list(os.listdir(tables_directory)))

    # add each plot to the tex file
    main_table = ""
    for table in tables:
        # \input{tables/high_level_ops_table_random walk_reads_d20_bs64}
        main_table += "\\input{\n" + f"{table_path}/{table.split(".tex")[0]}" + "}\n"

    f = open(f"{main_table_name}", "w+")
    f.write(main_table)
    f.close()

def parse_all_logs(
        logs_directory: str = "logs", 
        plots_directory: str = "plots",
        tables_directory: str = "tables",
        stats_to_plot: List[str] = ["read"],
        plot_path: str = "Figures", # path in the tex file
        table_path: str = "Tables", # path in the tex file
        main_plot_name: str = "main_plot.tex",
        main_table_name: str = "main_table.tex",
    ) -> None:
    all_plot_data = read_all_logs(logs_directory)
    average_and_scale_stats(all_plot_data)
    generate_all_plots_and_tables(plots_directory, tables_directory, stats_to_plot, all_plot_data)
    write_main_plot(plots_directory, plot_path, main_plot_name)
    write_main_table(tables_directory, table_path, main_table_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ldir", type=str, required=True)
    parser.add_argument("--pdir", type=str, required=True)
    parser.add_argument("--tdir", type=str, required=True)
    args = parser.parse_args()
    parse_all_logs(args.ldir, args.pdir, args.tdir)
