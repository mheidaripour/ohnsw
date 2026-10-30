#!/bin/bash

# Help function
show_help() {
    cat << EOF
Usage: $0 [INDEX_FILE] [QUERIES] [OPTIONS]

Description:
    Run oblivious HNSW search experiments with optional K-means clustering.
    This script extracts HNSW graph structure and performs oblivious search
    with round-trip counting and performance comparison against FAISS HNSW Search.

Parameters:
    INDEX_FILE      Path to HNSW index file (default: "data/1k/hnsw_32_40.index")
    QUERIES         Number of queries to execute (default: 100)

Options:
    --json FILE             Path to existing JSON graph file (skips extraction)
    --generate-json         Force re-extraction of JSON from index file
    --branching-factor N    Branching factor for oblivious out-tree (default: 2)
    --pointer-type TYPE     SmartPointer implementation: original, multiwrite,
                            multiwriterary, recursive (default: multiwrite)
    --bs N                  Block size in bytes; the multiwriterary pointer's
                            branching factor is derived from it (default: 64)
    --pointer-branching-factor N
                            Branching factor for the multiwriterary pointer
                            (default: derived from --bs)
    --path-oram-rary-read-evictions N
                            Full PathORAM evictions to run after each r-ary read
                            (default: pointer branching factor)
    --track-stash           Replay the PathORAM trace and print stash stats for
                            build and query phases
    --path-oram-tree-size N Number of PathORAM leaves for stash replay
    --path-oram-bucket-capacity N
                            PathORAM bucket capacity for stash replay (default: 4)
    --ef N                  ef_search parameter (default: 64)
    --k N                   Number of nearest neighbors (default: 10)
    --metric METRIC         Distance metric: 'ip' or 'l2' (default: 'l2')
    --cache-capacity N      Cache capacity in entries (default: 100)
    --cache-eviction PCT    Cache eviction percentage (default: 0.2)
    --partitioning METHOD   Base-layer partitioning: none, kmeans, metis, louvain,
                            singleton (default: none). Preferred over the legacy
                            flags below.
    --n-clusters N          Number of clusters/partitions for kmeans/metis
                            (default: auto, sqrt(ntotal)). Ignored by singleton,
                            which always uses one partition per node.
    --use-kmeans            Enable K-means clustering (default: disabled) [legacy]
    --use-louvain           Enable Louvain clustering (default: disabled) [legacy]
    --non-oblivious-upper   Use non-oblivious upper layers (default: oblivious)
    --make-copies           Toggles if extra copies are creating during OGraph operations (default: disabled)
    -h, --help              Show this help message and exit

    --partitioning takes precedence over --use-kmeans/--use-louvain.
    If kmeans and louvain are both enabled, kmeans takes priority.
    METIS requires pymetis (pip install pymetis); it falls back to
    round-robin partitioning if pymetis is present but part_graph fails.

JSON File Handling:
    By default, the script uses data/1k/hnsw_32_40.json if it exists.
    
    --json FILE         Use specific JSON file (skips extraction)
    --generate-json     Force re-extraction even if JSON exists
    
    If no JSON file exists and --json not specified, extracts from INDEX_FILE.

Examples:
    # Use default JSON (data/1k/hnsw_32_40.json), no extraction
    $0

    # Use existing custom JSON file
    $0 --json data/1k/my_graph.json

    # Force re-extraction from index
    $0 data/1k/hnsw_32_40.index 100 --generate-json

    # With K-means clustering and custom cache
    $0 data/1k/hnsw_32_40.index 100 --use-kmeans --cache-capacity 50

    # Full custom configuration
    $0 data/1k/hnsw_32_40.index 100 \\
       --json data/1k/custom.json \\
       --use-kmeans \\
       --metric l2 \\
       --cache-capacity 15 \\
       --ef 64

Prerequisites:
    - Build directory must exist with extract_hnsw binary (for extraction)
    - Python environment with faiss-cpu and scikit-learn
    - HNSW index file or JSON graph file must exist

Output:
    - Creates JSON graph file if extraction performed
    - Displays clustering statistics and round-trip counts
    - Shows performance comparison between oblivious and FAISS search
    - Creates cluster_access_log_*.csv if K-means clustering enabled

EOF
}

# Check for help flag
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_help
    exit 0
fi

# Default values
DEFAULT_INDEX="data/1k/hnsw_32_40.index"
DEFAULT_QUERIES="100"
DEFAULT_JSON="data/1k/hnsw_32_40.json"

# Initialize with defaults
INDEX_FILE="$DEFAULT_INDEX"
QUERIES="$DEFAULT_QUERIES"

# Parse optional arguments
BRANCHING_FACTOR="2"
POINTER_TYPE="multiwrite"
BLOCK_SIZE="64"
POINTER_BRANCHING_FACTOR=""
PATH_ORAM_RARY_READ_EVICTIONS=""
TRACK_STASH=""
PATH_ORAM_TREE_SIZE=""
PATH_ORAM_BUCKET_CAPACITY="4"
EF_SEARCH="64"
K_NEIGHBORS="10"
METRIC="l2"
CACHE_CAPACITY="100"
CACHE_EVICTION="0.2"
PARTITIONING=""
N_CLUSTERS=""
USE_KMEANS=""
USE_LOUVAIN=""
NON_OBLIVIOUS_UPPER=""
MAKE_COPIES=""
JSON_FILE=""
GENERATE_JSON=false

# Check if first argument is a file path or an option
if [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; then
    INDEX_FILE="$1"
    shift
    
    # Check if second argument is a number or an option
    if [[ $# -gt 0 ]] && [[ "$1" =~ ^[0-9]+$ ]]; then
        QUERIES="$1"
        shift
    fi
fi

# Parse named arguments (no shift needed, already shifted above)
while [[ $# -gt 0 ]]; do
    case $1 in
        --json)
            JSON_FILE="$2"
            shift 2
            ;;
        --generate-json)
            GENERATE_JSON=true
            shift
            ;;
        --branching-factor)
            BRANCHING_FACTOR="$2"
            shift 2
            ;;
        --pointer-type)
            POINTER_TYPE="$2"
            shift 2
            ;;
        --bs)
            BLOCK_SIZE="$2"
            shift 2
            ;;
        --pointer-branching-factor)
            POINTER_BRANCHING_FACTOR="$2"
            shift 2
            ;;
        --path-oram-rary-read-evictions|--rary-read-evictions)
            PATH_ORAM_RARY_READ_EVICTIONS="$2"
            shift 2
            ;;
        --track-stash|--track-stack-size)
            TRACK_STASH="--track-stash"
            shift
            ;;
        --path-oram-tree-size)
            PATH_ORAM_TREE_SIZE="$2"
            shift 2
            ;;
        --path-oram-bucket-capacity)
            PATH_ORAM_BUCKET_CAPACITY="$2"
            shift 2
            ;;
        --ef)
            EF_SEARCH="$2"
            shift 2
            ;;
        --k)
            K_NEIGHBORS="$2"
            shift 2
            ;;
        --metric)
            METRIC="$2"
            shift 2
            ;;
        --cache-capacity)
            CACHE_CAPACITY="$2"
            shift 2
            ;;
        --cache-eviction)
            CACHE_EVICTION="$2"
            shift 2
            ;;
        --partitioning)
            PARTITIONING="$2"
            shift 2
            ;;
        --n-clusters)
            N_CLUSTERS="$2"
            shift 2
            ;;
        --use-kmeans)
            USE_KMEANS="--use-kmeans-clustering"
            shift
            ;;
        --use-louvain)
            USE_LOUVAIN="--use-louvain-clustering"
            shift
            ;;
        --non-oblivious-upper)
            NON_OBLIVIOUS_UPPER="--non-oblivious-upper"
            shift
            ;;
        --make-copies)
            MAKE_COPIES="--make-copies"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use '$0 --help' for usage information."
            exit 1
            ;;
    esac
done

# Validate before the (potentially slow) extraction step, so a typo fails fast
# instead of after JSON extraction has already run.
if [[ -n "$PARTITIONING" ]]; then
    case "$PARTITIONING" in
        none|kmeans|metis|louvain|singleton) ;;
        *)
            echo "Error: invalid --partitioning '$PARTITIONING'"
            echo "Choose one of: none, kmeans, metis, louvain, singleton"
            exit 1
            ;;
    esac
fi

if [[ -n "$N_CLUSTERS" ]] && ! [[ "$N_CLUSTERS" =~ ^[0-9]+$ ]]; then
    echo "Error: --n-clusters must be a positive integer, got '$N_CLUSTERS'"
    exit 1
fi

for numeric_arg in \
    "branching factor:$BRANCHING_FACTOR" \
    "block size:$BLOCK_SIZE" \
    "pointer branching factor:$POINTER_BRANCHING_FACTOR" \
    "r-ary read evictions:$PATH_ORAM_RARY_READ_EVICTIONS" \
    "PathORAM tree size:$PATH_ORAM_TREE_SIZE" \
    "PathORAM bucket capacity:$PATH_ORAM_BUCKET_CAPACITY"
do
    label="${numeric_arg%%:*}"
    value="${numeric_arg#*:}"
    if [[ -n "$value" ]] && ! [[ "$value" =~ ^[0-9]+$ ]]; then
        echo "Error: $label must be a positive integer, got '$value'"
        exit 1
    fi
    if [[ -n "$value" ]] && (( value <= 0 )); then
        echo "Error: $label must be a positive integer, got '$value'"
        exit 1
    fi
done

if [[ -n "$TRACK_STASH" ]] && [[ -z "$PATH_ORAM_TREE_SIZE" ]]; then
    echo "Error: --track-stash requires --path-oram-tree-size"
    exit 1
fi

# Determine JSON file to use
if [[ -n "$JSON_FILE" ]]; then
    # User specified JSON file explicitly
    if [[ ! -f "$JSON_FILE" ]]; then
        echo "Error: Specified JSON file '$JSON_FILE' not found!"
        exit 1
    fi
    SKIP_EXTRACTION=true
    echo "Using specified JSON file: $JSON_FILE"
elif [[ "$GENERATE_JSON" == true ]]; then
    # Force re-extraction
    JSON_FILE="${INDEX_FILE%.*}.json"
    SKIP_EXTRACTION=false
    echo "Forcing re-extraction to: $JSON_FILE"
elif [[ -f "$DEFAULT_JSON" ]]; then
    # Use default JSON if it exists
    JSON_FILE="$DEFAULT_JSON"
    SKIP_EXTRACTION=true
    echo "Using default JSON file: $JSON_FILE"
else
    # Extract from index file
    JSON_FILE="${INDEX_FILE%.*}.json"
    SKIP_EXTRACTION=false
    echo "Will extract JSON from index: $INDEX_FILE → $JSON_FILE"
fi

# Print configuration
echo ""
echo "=== Oblivious HNSW Experiment ==="
echo "Index file:         $INDEX_FILE"
echo "JSON file:          $JSON_FILE"
echo "Queries:            $QUERIES"
echo "Branching factor:   $BRANCHING_FACTOR"
echo "Pointer type:       $POINTER_TYPE"
echo "Pointer block size: $BLOCK_SIZE"
echo "Pointer branching:  ${POINTER_BRANCHING_FACTOR:-derived from block size}"
echo "R-ary read evicts:  ${PATH_ORAM_RARY_READ_EVICTIONS:-pointer branching factor}"
echo "Stash tracking:     $([ -n "$TRACK_STASH" ] || [ -n "$PATH_ORAM_TREE_SIZE" ] && echo "ENABLED" || echo "DISABLED")"
if [[ -n "$TRACK_STASH" || -n "$PATH_ORAM_TREE_SIZE" ]]; then
    echo "PathORAM leaves:    $PATH_ORAM_TREE_SIZE"
    echo "PathORAM bucket Z:  $PATH_ORAM_BUCKET_CAPACITY"
fi
echo "ef_search:          $EF_SEARCH"
echo "k neighbors:        $K_NEIGHBORS"
echo "Metric:             $METRIC"
echo "Cache capacity:     $CACHE_CAPACITY"
echo "Cache eviction:     $CACHE_EVICTION"
echo "Partitioning:       ${PARTITIONING:-none (using legacy flags below)}"
echo "Num clusters:       ${N_CLUSTERS:-auto (sqrt(ntotal))}"
echo "K-means clustering: $([ -n "$USE_KMEANS" ] && echo "ENABLED" || echo "DISABLED")"
echo "Louvain clustering: $([ -n "$USE_LOUVAIN" ] && echo "ENABLED" || echo "DISABLED")"
echo "Upper layers:       $([ -n "$NON_OBLIVIOUS_UPPER" ] && echo "NON-OBLIVIOUS" || echo "OBLIVIOUS")"
echo "Make copies:        $([ -n "$MAKE_COPIES" ] && echo "ENABLED" || echo "DISABLED")"
echo "=================================="

# Validate inputs
if ! [[ "$QUERIES" =~ ^[0-9]+$ ]] || [ "$QUERIES" -le 0 ]; then
    echo "Error: QUERIES must be a positive integer, got '$QUERIES'"
    echo "Use '$0 --help' for usage information."
    exit 1
fi

# Extract HNSW index if needed
if [[ "$SKIP_EXTRACTION" == false ]]; then
    # Check prerequisites
    if [[ ! -f "./build/extract_hnsw" ]]; then
        echo "Error: extract_hnsw binary not found in ./build/"
        echo "Run 'cmake --build build' to compile first."
        exit 1
    fi
    
    # Validate index file exists
    if [[ ! -f "$INDEX_FILE" ]]; then
        echo "Error: Index file '$INDEX_FILE' not found!"
        echo "Use '$0 --help' for usage information."
        exit 1
    fi
    
    echo ""
    echo "Step 1: Extracting HNSW graph structure..."
    ./build/extract_hnsw "${INDEX_FILE}" "${JSON_FILE}"
    
    if [[ $? -ne 0 ]]; then
        echo "Error: Failed to extract HNSW graph"
        exit 1
    fi
    echo "✓ JSON extracted to: $JSON_FILE"
else
    echo ""
    echo "Step 1: Skipping extraction (using existing JSON)"
fi

# Validate JSON file exists
if [[ ! -f "$JSON_FILE" ]]; then
    echo "Error: JSON file '$JSON_FILE' not found after extraction!"
    exit 1
fi

# Run Python script
echo ""
echo "Step 2: Running oblivious HNSW search..."

# Every option is forwarded on every path. Previously the kmeans, louvain and
# standard branches each passed a different subset, so e.g. --metric/--ef/--k
# were silently dropped unless kmeans was enabled, while the banner above still
# reported them as if they had applied.
O_HNSW_ARGS=(
    --graph "${JSON_FILE}"
    --faiss-index "${INDEX_FILE}"
    --q "${QUERIES}"
    --ef "${EF_SEARCH}"
    --k "${K_NEIGHBORS}"
    --metric "${METRIC}"
    --cache-capacity "${CACHE_CAPACITY}"
    --cache-eviction "${CACHE_EVICTION}"
    --branching-factor "${BRANCHING_FACTOR}"
    --pointer-type "${POINTER_TYPE}"
    --bs "${BLOCK_SIZE}"
    --path-oram-bucket-capacity "${PATH_ORAM_BUCKET_CAPACITY}"
)

# Optional switches: appended only when set, so no empty arguments are passed.
# --partitioning takes precedence over the legacy --use-* flags in o_hnsw.py;
# if both legacy flags are given, o_hnsw.py gives kmeans priority.
[[ -n "$PARTITIONING" ]] && O_HNSW_ARGS+=(--partitioning "$PARTITIONING")
[[ -n "$N_CLUSTERS" ]] && O_HNSW_ARGS+=(--n-clusters "$N_CLUSTERS")
[[ -n "$POINTER_BRANCHING_FACTOR" ]] && O_HNSW_ARGS+=(--pointer-branching-factor "$POINTER_BRANCHING_FACTOR")
[[ -n "$PATH_ORAM_RARY_READ_EVICTIONS" ]] && O_HNSW_ARGS+=(--path-oram-rary-read-evictions "$PATH_ORAM_RARY_READ_EVICTIONS")
[[ -n "$PATH_ORAM_TREE_SIZE" ]] && O_HNSW_ARGS+=(--path-oram-tree-size "$PATH_ORAM_TREE_SIZE")
[[ -n "$TRACK_STASH" ]] && O_HNSW_ARGS+=("$TRACK_STASH")
[[ -n "$USE_KMEANS" ]] && O_HNSW_ARGS+=("$USE_KMEANS")
[[ -n "$USE_LOUVAIN" ]] && O_HNSW_ARGS+=("$USE_LOUVAIN")
[[ -n "$NON_OBLIVIOUS_UPPER" ]] && O_HNSW_ARGS+=("$NON_OBLIVIOUS_UPPER")
[[ -n "$MAKE_COPIES" ]] && O_HNSW_ARGS+=("$MAKE_COPIES")

python3 o_hnsw.py "${O_HNSW_ARGS[@]}"

if [[ $? -ne 0 ]]; then
    echo "Error: Python script failed"
    exit 1
fi

echo ""
echo "=== Experiment Complete ==="
