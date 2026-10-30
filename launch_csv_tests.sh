#!/bin/bash

# graph_name="emailEucore"
# dataset_file="emailEucore.txt"
# graph_name="p2pGnutella04"
# dataset_file="p2pGnutella04.txt"
# graph_name="gpluscombined"
# dataset_file="gpluscombined.txt"
# graph_name="roadNetPA"
# dataset_file="roadNetPA.txt"
# graph_name="webBerkStan"
# dataset_file="webBerkStan.txt"
graph_name="twitchgamers"
dataset_file="twitchgamers.csv"
directed=false

case "$directed" in
  true) directed_line="--directed" ;;
  false) directed_line="--undirected" ;; 
esac

LOG_DIR="real-dataset-tests/$graph_name/logs"
mkdir -p "$LOG_DIR"

file="real-dataset-tests/$graph_name/$dataset_file"
block_sizes=(64 4096)
trials=50
algorithms=("rw" "cd" "pr") # "dtc"
pointers=("original" "multiwrite" "multiwriterary" "recursive")
make_copies_plural=(false)
prime_graph_plural=(true)
static_insertion_plural=(true) 
walk_length=50
damping_factor=0.9
default_branching_factor=64
small_block_branching_factor=6
path_oram_rary_read_evictions=1
path_oram_tree_size=1073741824
path_oram_bucket_capacity=4

for static_insertion in "${static_insertion_plural[@]}"; do
  case "$static_insertion" in
    true) static_line="--staticinsertion" ;;
    false) static_line="--no_staticinsertion" ;; 
  esac

  for prime_graph in "${prime_graph_plural[@]}"; do
    case "$prime_graph" in
      true) prime_line="--prime" ;;
      false) prime_line="--no_prime" ;; 
    esac

    for make_copies in "${make_copies_plural[@]}"; do
      case "$make_copies" in
        true) copies_line="--copies" ;;
        false) copies_line="--no_copies" ;; 
      esac
    
      for bs in "${block_sizes[@]}"; do
        for pointer in "${pointers[@]}"; do
          case "$pointer" in
            "recursive") move_semantics_plural=(true) ;;
            "original")    move_semantics_plural=(true false) ;; 
            "multiwrite")    move_semantics_plural=(true) ;;
            "multiwriterary")    move_semantics_plural=(true) ;;
            *)       continue ;;
          esac

          for move_semantics in "${move_semantics_plural[@]}"; do
            case "$move_semantics" in
              true) move_line="--move" ;;
              false) move_line="--no_move" ;; 
            esac
            for algorithm in "${algorithms[@]}"; do
              case "$algorithm" in
                "rw") 
                  extra_info="_wl-${walk_length}"
                  ;;
                "pr") 
                  extra_info="_wl-${walk_length}_df-${damping_factor}"
                  ;; 
                *) 
                  extra_info=""
                  ;;
              esac

              rary_info=""
              rary_args=()
              if [ "$pointer" = "multiwriterary" ]; then
                branching_factor="$default_branching_factor"
                if [ "$bs" -eq 64 ]; then
                  branching_factor="$small_block_branching_factor"
                fi
                rary_info="_bf-${branching_factor}_raryevict-${path_oram_rary_read_evictions}"
                rary_args=(
                  --branching_factor "$branching_factor"
                  --path_oram_rary_read_evictions "$path_oram_rary_read_evictions"
                )
              fi

              log_file="$LOG_DIR/${graph_name}_bs-${bs}_trials-${trials}_pt-${pointer}_alg-${algorithm}_move-${move_semantics}_copies-${make_copies}_prime-${prime_graph}_staticinsertion-${static_insertion}${extra_info}${rary_info}.log"
              echo "Launching test: --graph_name $graph_name $directed_line --file $file --bs $bs --trials $trials --pt $pointer --alg $algorithm $move_line $copies_line $prime_line $static_line --wl $walk_length --df $damping_factor --path_oram_tree_size $path_oram_tree_size --path_oram_bucket_capacity $path_oram_bucket_capacity ${rary_args[*]}"
              python3 python-implementation/benchmark_graph_from_csv.py \
                --graph_name "$graph_name" \
                "$directed_line" \
                --file "$file" \
                --bs "$bs" \
                --trials "$trials" \
                --pt "$pointer" \
                --alg "$algorithm" \
                "$move_line" \
                "$copies_line" \
                "$prime_line" \
                "$static_line" \
                --wl "$walk_length" \
                --df "$damping_factor" \
                --path_oram_tree_size "$path_oram_tree_size" \
                --path_oram_bucket_capacity "$path_oram_bucket_capacity" \
                "${rary_args[@]}" \
                > "$log_file" 2>&1 &
            
            done
          done
        done
      done
    done
  done
done

echo "All jobs launched in background. Use 'jobs' or 'ps' to monitor."
