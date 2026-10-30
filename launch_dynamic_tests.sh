#!/bin/bash

LOG_DIR="local-tests/logs"
mkdir -p "$LOG_DIR"

block_sizes=(64 4096)
trials=50
degrees=(20 100)
powers=$(seq 5 20)
algorithms=("rw" "cd" "pr") # "dtc"
pointers=("original" "multiwrite" "recursive")
make_copies_plural=(false)
prime_graph_plural=(true)
static_insertion_plural=(true) 
walk_length=50
damping_factor=0.9

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
        for power in $powers; do
          n=$((2**power))
          for d in "${degrees[@]}"; do
            if [ "$d" -le "$n" ]; then
              for pointer in "${pointers[@]}"; do
                case "$pointer" in
                  "recursive") move_semantics_plural=(true) ;;
                  "original")    move_semantics_plural=(true false) ;; 
                  "multiwrite")    move_semantics_plural=(true) ;;
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

                    log_file="$LOG_DIR/ER_n-${n}_d-${d}_bs-${bs}_trials-${trials}_pt-${pointer}_alg-${algorithm}_move-${move_semantics}_copies-${make_copies}_prime-${prime_graph}_staticinsertion-${static_insertion}${extra_info}.log"
                    echo "Launching test: --n $n --d $d --bs $bs --trials $trials --pt $pointer --alg $algorithm $move_line $copies_line $prime_line $static_line --wl $walk_length --df $damping_factor --dynamic_tests"
                    python3 python-implementation/benchmark_single_alg.py \
                      --n "$n" \
                      --d "$d" \
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
                      --dynamic_tests \
                      > "$log_file" 2>&1 &
                  
                  done
                done
              done
            fi
          done
        done
      done
    done
  done
done

echo "All jobs launched in background. Use 'jobs' or 'ps' to monitor."
