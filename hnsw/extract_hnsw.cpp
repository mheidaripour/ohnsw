#include <faiss/Index.h>
#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>
#include <faiss/IndexIDMap.h>
#include <faiss/IndexRefine.h>

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// --- Minimal JSON helpers (escape + write arrays/objects) ---
static std::string json_escape(const std::string& s) {
    std::ostringstream o;
    o << '"';
    for (char c : s) {
        switch (c) {
            case '"':  o << "\\\""; break;
            case '\\': o << "\\\\"; break;
            case '\b': o << "\\b";  break;
            case '\f': o << "\\f";  break;
            case '\n': o << "\\n";  break;
            case '\r': o << "\\r";  break;
            case '\t': o << "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    o << "\\u" << std::hex << std::uppercase
                      << std::setw(4) << std::setfill('0') << (int)(unsigned char)c
                      << std::nouppercase << std::dec;
                } else {
                    o << c;
                }
        }
    }
    o << '"';
    return o.str();
}

static void write_int_array(std::ostream& os, const std::vector<int64_t>& v) {
    os << '[';
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) os << ',';
        os << v[i];
    }
    os << ']';
}

static void write_int_array(std::ostream& os, const std::vector<int32_t>& v) {
    os << '[';
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) os << ',';
        os << v[i];
    }
    os << ']';
}

static void write_size_t_array(std::ostream& os, const std::vector<size_t>& v) {
    os << '[';
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) os << ',';
        os << v[i];
    }
    os << ']';
}

// Try to unwrap common wrappers to reach an IndexHNSW-derived core.
static faiss::Index* unwrap_to_core(faiss::Index* idx) {
    if (auto irf = dynamic_cast<faiss::IndexRefineFlat*>(idx)) {
        return unwrap_to_core(irf->base_index);
    }
    if (auto ir  = dynamic_cast<faiss::IndexRefine*>(idx)) {
        return unwrap_to_core(ir->base_index);
    }
    if (auto idm = dynamic_cast<faiss::IndexIDMap*>(idx)) {
        return unwrap_to_core(idm->index);
    }
    return idx;
}

struct LayerStats {
    int layer = 0;
    int64_t nodes = 0;
    int64_t edges_directed = 0;     // sum of out-degrees (neighbor list sizes)
    int32_t min_deg = std::numeric_limits<int32_t>::max();
    int32_t max_deg = 0;
    double avg_deg = 0.0;
    double p90_deg = 0.0;           // 90th percentile (approx)
};

static std::vector<LayerStats> compute_layer_stats(const faiss::HNSW& H, int64_t ntotal) {
    const int maxL = H.max_level;            // topmost layer index
    std::vector<LayerStats> stats(maxL + 1);
    for (int l = 0; l <= maxL; ++l) stats[l].layer = l;

    std::vector<int32_t> degs; degs.reserve(1024);

    for (int l = 0; l <= maxL; ++l) {
        degs.clear();
        int64_t nodes_here = 0;
        int64_t edges_here = 0;
        int32_t min_deg = std::numeric_limits<int32_t>::max();
        int32_t max_deg = 0;

        for (int64_t i = 0; i < ntotal; ++i) {
            // Node i participates on layer l iff levels[i] > l (FAISS uses count of layers per node)
            if (i >= (int64_t)H.levels.size()) continue;
            if (H.levels[i] <= l) continue;

            size_t begin = 0, end = 0;
            H.neighbor_range((faiss::idx_t)i, l, &begin, &end);

            // Degree = number of stored neighbors in this layer slice.
            int32_t deg = static_cast<int32_t>(end - begin);

            // Some builds might store -1 as fillers; filter them out if present.
            // (neighbor_range usually returns the used slice; still robustify.)
            if (deg > 0) {
                int32_t effective = 0;
                for (size_t p = begin; p < end; ++p) {
                    if (H.neighbors[p] >= 0) ++effective;
                }
                deg = effective;
            }

            ++nodes_here;
            edges_here += deg;
            min_deg = std::min(min_deg, deg);
            max_deg = std::max(max_deg, deg);
            degs.push_back(deg);
        }

        double avg = nodes_here ? double(edges_here) / double(nodes_here) : 0.0;

        double p90 = 0.0;
        if (!degs.empty()) {
            std::nth_element(degs.begin(),
                             degs.begin() + (degs.size() * 9) / 10,
                             degs.end());
            p90 = double(degs[(degs.size() * 9) / 10]);
        }

        if (nodes_here == 0) {
            min_deg = 0;
            max_deg = 0;
        }

        stats[l].nodes = nodes_here;
        stats[l].edges_directed = edges_here;
        stats[l].min_deg = min_deg;
        stats[l].max_deg = max_deg;
        stats[l].avg_deg = avg;
        stats[l].p90_deg = p90;
    }
    return stats;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <faiss_index_file> <output.json>\n";
        return 1;
    }
    const std::string index_path = argv[1];
    const std::string out_path   = argv[2];

    try {
        std::unique_ptr<faiss::Index> idx(faiss::read_index(index_path.c_str()));
        if (!idx) {
            std::cerr << "Failed to read index from: " << index_path << "\n";
            return 2;
        }

        faiss::Index* core = unwrap_to_core(idx.get());
        auto* hnsw_idx = dynamic_cast<faiss::IndexHNSW*>(core);
        if (!hnsw_idx) {
            std::cerr << "Index is not an HNSW family index after unwrap.\n";
            return 3;
        }

        const auto& H = hnsw_idx->hnsw;
        const int64_t ntotal = hnsw_idx->ntotal;

        // ---- Compute per-layer stats and print them ----
        auto stats = compute_layer_stats(H, ntotal);

        std::cout << "HNSW Graph Layer Stats\n";
        std::cout << "entry_point=" << H.entry_point
                  << " max_level=" << H.max_level
                  << " ntotal=" << ntotal << "\n";
        std::cout << std::fixed << std::setprecision(3);
        for (const auto& s : stats) {
            std::cout << "  Layer " << s.layer
                      << " | nodes=" << s.nodes
                      << " edges(dir)=" << s.edges_directed
                      << " avg_deg=" << s.avg_deg
                      << " min_deg=" << s.min_deg
                      << " max_deg=" << s.max_deg
                      << " p90_deg=" << s.p90_deg
                      << "\n";
        }

        // ---- Export JSON (including the stats) ----
        std::ofstream os(out_path);
        if (!os) {
            std::cerr << "Cannot open output: " << out_path << "\n";
            return 4;
        }

        os << "{\n";
        // meta
        os << "  \"meta\": {\n";
        os << "    \"ntotal\": " << ntotal << ",\n";
        os << "    \"d\": " << hnsw_idx->d << ",\n";
        os << "    \"metric_type\": " << int(hnsw_idx->metric_type) << ",\n";
        os << "    \"entry_point\": " << H.entry_point << ",\n";
        os << "    \"max_level\": " << H.max_level << ",\n";
        os << "    \"efConstruction\": " << H.efConstruction << ",\n";
        os << "    \"efSearch\": " << H.efSearch << "\n";
        os << "  },\n";

        // raw tables
        os << "  \"tables\": {\n";
        os << "    \"levels\": ";
        { std::vector<int32_t> lvls(H.levels.begin(), H.levels.end());
          write_int_array(os, lvls); os << ",\n"; }
        os << "    \"offsets\": ";
        { std::vector<size_t> offs(H.offsets.begin(), H.offsets.end());
          write_size_t_array(os, offs); os << ",\n"; }
        os << "    \"neighbors\": ";
        { std::vector<int32_t> neigh(H.neighbors.begin(), H.neighbors.end());
          write_int_array(os, neigh); os << "\n"; }
        os << "  },\n";

        // layer_stats
        os << "  \"layer_stats\": [\n";
        for (size_t i = 0; i < stats.size(); ++i) {
            const auto& s = stats[i];
            os << "    {"
               << "\"layer\":" << s.layer
               << ",\"nodes\":" << s.nodes
               << ",\"edges_directed\":" << s.edges_directed
               << ",\"avg_deg\":" << std::setprecision(6) << s.avg_deg
               << ",\"min_deg\":" << s.min_deg
               << ",\"max_deg\":" << s.max_deg
               << ",\"p90_deg\":" << s.p90_deg
               << "}";
            if (i + 1 < stats.size()) os << ",";
            os << "\n";
        }
        os << "  ],\n";

        // graph (per-node, per-layer neighbors)
        os << "  \"graph\": [\n";
        for (int64_t i = 0; i < ntotal; ++i) {
            if (i) os << ",\n";
            os << "    {\n";
            os << "      \"node\": " << i << ",\n";
            os << "      \"levels\": [\n";

            int node_levels = (i < (int64_t)H.levels.size()) ? H.levels[i] : 0;
            for (int l = 0; l < node_levels; ++l) {
                if (l) os << ",\n";
                size_t begin = 0, end = 0;
                H.neighbor_range((faiss::idx_t)i, l, &begin, &end);
                os << "        {\"layer\": " << l << ", \"neighbors\": [";
                bool first = true;
                for (size_t p = begin; p < end; ++p) {
                    int n = H.neighbors[p];
                    if (n < 0) continue; // skip unused slots if any
                    if (!first) os << ',';
                    os << n;
                    first = false;
                }
                os << "]}";
            }
            os << "\n      ]\n";
            os << "    }";
        }
        os << "\n  ]\n";

        os << "}\n";
        os.close();

        std::cout << "Exported HNSW graph + layer stats to " << out_path << "\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 5;
    }
}
