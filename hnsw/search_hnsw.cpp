#include "search_hnsw.hpp"
#include <nlohmann/json.hpp>

#include <faiss/Index.h>
#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <unordered_set>

static std::vector<float> load_fvecs(const std::string& path, int& d_out, int64_t& n_out) {
    d_out = 0; n_out = 0;
    std::ifstream is(path, std::ios::binary);
    if (!is) throw std::runtime_error("cannot open fvecs: " + path);

    int32_t d = 0;
    while (true) {
        int32_t dim;
        if (!is.read(reinterpret_cast<char*>(&dim), sizeof(int32_t))) break;
        if (d == 0) d = dim;
        else if (dim != d) throw std::runtime_error("mixed dims in fvecs: " + path);
        is.seekg(sizeof(float) * dim, std::ios::cur);
        ++n_out;
    }
    if (n_out == 0) return {};
    d_out = d;

    is.clear();
    is.seekg(0, std::ios::beg);
    std::vector<float> data(static_cast<size_t>(n_out) * d);
    for (int64_t i = 0; i < n_out; ++i) {
        int32_t dim;
        is.read(reinterpret_cast<char*>(&dim), sizeof(int32_t));
        if (dim != d) throw std::runtime_error("dim mismatch in fvecs second pass");
        is.read(reinterpret_cast<char*>(data.data() + i * d), sizeof(float) * d);
    }
    return data;
}

static void load_hnsw_graph_json(const std::string& json_path, HNSWView& G) {
    std::ifstream is(json_path);
    if (!is) throw std::runtime_error("cannot open graph json: " + json_path);
    nlohmann::json j; is >> j;

    const auto& meta = j.at("meta");
    const auto& graph = j.at("graph");

    const int64_t ntotal = meta.at("ntotal").get<int64_t>();
    const int max_level   = meta.at("max_level").get<int>();
    const int entry_point = meta.at("entry_point").get<int>();

    G.ntotal = ntotal;
    G.max_level = max_level;
    G.entry_point = entry_point;
    G.adjacency.assign(max_level + 1, std::vector<std::vector<int>>(ntotal));
    G.node_levels.assign(ntotal, 0);

    for (const auto& node_obj : graph) {
        int i = node_obj.at("node").get<int>();
        const auto& lv = node_obj.at("levels");
        int maxL_for_node = 0;
        for (const auto& layer_obj : lv) {
            int L = layer_obj.at("layer").get<int>();
            const auto& neigh = layer_obj.at("neighbors");
            auto& dst = G.adjacency[L][i];
            dst.reserve(neigh.size());
            for (const auto& n : neigh) dst.push_back(n.get<int>());
            maxL_for_node = std::max(maxL_for_node, L);
        }
        G.node_levels[i] = maxL_for_node + 1;
    }
}

static std::vector<std::pair<int, float>>
faiss_search_topk(faiss::Index* index, const float* q, int k, int ef_search) {
    if (auto* hnsw = dynamic_cast<faiss::IndexHNSW*>(index)) {
        hnsw->hnsw.efSearch = ef_search;
    } else if (auto* hnswF = dynamic_cast<faiss::IndexHNSWFlat*>(index)) {
        hnswF->hnsw.efSearch = ef_search;
    }
    std::vector<faiss::idx_t> I(k);
    std::vector<float> D(k);
    index->search(1, q, k, D.data(), I.data());
    std::vector<std::pair<int, float>> out;
    out.reserve(k);
    for (int i = 0; i < k; ++i) out.emplace_back(static_cast<int>(I[i]), D[i]);
    return out;
}

static double recall_at_k(const std::vector<Neighbor>& A,
                          const std::vector<std::pair<int,float>>& B) {
    std::unordered_set<int> idsB;
    idsB.reserve(B.size());
    for (auto& p : B) idsB.insert(p.first);
    if (A.empty()) return 0.0;
    int hit = 0;
    for (auto& n : A) if (idsB.count(n.id)) ++hit;
    return double(hit) / double(A.size());
}

static bool exact_match_ids(const std::vector<Neighbor>& A,
                            const std::vector<std::pair<int,float>>& B) {
    if (A.size() != B.size()) return false;
    for (size_t i = 0; i < A.size(); ++i) {
        if (A[i].id != B[i].first) return false;
    }
    return true;
}

static void usage(const char* prog) {
    std::cerr <<
        "Usage:\n"
        "  " << prog << " --graph graph.json --base data/base.fvecs --queries data/query.fvecs \\\n"
        "               [--k 10] [--ef 64] [--metric l2|ip] [--limit N] \\\n"
        "               [--faiss-index path.index]\n"
        "\n"
        "Prints top-k ids + distances for (up to) 10 queries. If --faiss-index is given,\n"
        "also runs FAISS native search and reports agreement metrics.\n";
}

int main(int argc, char** argv) {
    std::string graph_path, base_path, query_path;
    std::string faiss_index_path;
    int k = 10, ef = 64, limit = -1;
    MetricType metric = MetricType::L2;
    bool print_results = true;


    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--graph") && i + 1 < argc) graph_path = argv[++i];
        else if (!std::strcmp(argv[i], "--base") && i + 1 < argc) base_path = argv[++i];
        else if (!std::strcmp(argv[i], "--queries") && i + 1 < argc) query_path = argv[++i];
        else if (!std::strcmp(argv[i], "--k") && i + 1 < argc) k = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--ef") && i + 1 < argc) ef = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--limit") && i + 1 < argc) limit = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--metric") && i + 1 < argc) {
            ++i;
            if (!std::strcmp(argv[i], "l2")) metric = MetricType::L2;
            else if (!std::strcmp(argv[i], "ip")) metric = MetricType::InnerProduct;
            else { std::cerr << "Unknown metric: " << argv[i] << "\n"; return 1; }
        } else if (!std::strcmp(argv[i], "--faiss-index") && i + 1 < argc) {
            faiss_index_path = argv[++i];
        } else if (!std::strcmp(argv[i], "--no-print")) {
            print_results = false;
        }
        else {
            usage(argv[0]); return 1;
        }
    }

    if (graph_path.empty() || base_path.empty() || query_path.empty()) {
        usage(argv[0]); return 1;
    }

    try {
        HNSWView G;
        load_hnsw_graph_json(graph_path, G);

        int d_base = 0; int64_t n_base = 0;
        auto base = load_fvecs(base_path, d_base, n_base);
        if (n_base != G.ntotal) {
            std::cerr << "WARNING: base.fvecs vectors (" << n_base
                      << ") != graph.ntotal (" << G.ntotal << ")\n";
            if (n_base < G.ntotal) throw std::runtime_error("Dataset smaller than graph.ntotal");
        }
        MatrixViewF32 X{ base.data(), n_base, d_base };

        int d_q = 0; int64_t nq = 0;
        auto queries = load_fvecs(query_path, d_q, nq);
        if (d_q != X.d) {
            throw std::runtime_error("dimension mismatch: queries d = " + std::to_string(d_q) +
                                     " vs base d = " + std::to_string(X.d));
        }

        SearchParams sp;
        sp.metric = metric;
        sp.ef_search = std::max(ef, k);

        std::unique_ptr<faiss::Index> faiss_index_ptr;
        if (!faiss_index_path.empty()) {
            faiss::Index* idx = faiss::read_index(faiss_index_path.c_str());
            if (!idx) throw std::runtime_error("failed to read faiss index: " + faiss_index_path);

            bool metric_ok = true;
            if (metric == MetricType::L2 && idx->metric_type != faiss::MetricType::METRIC_L2) metric_ok = false;
            if (metric == MetricType::InnerProduct && idx->metric_type != faiss::MetricType::METRIC_INNER_PRODUCT) metric_ok = false;
            if (!metric_ok) {
                std::cerr << "WARNING: CLI metric and FAISS index metric differ (CLI="
                          << (metric == MetricType::L2 ? "L2" : "IP")
                          << ", FAISS=" << (idx->metric_type == faiss::METRIC_L2 ? "L2" : "IP")
                          << ")\n";
            }
            if (idx->d != d_q) {
                throw std::runtime_error("FAISS index dim != query dim");
            }
            faiss_index_ptr.reset(idx);
        }

        const int64_t run_nq = (limit > 0) ? std::min<int64_t>(nq, limit) : nq;
        // constexpr int kMaxQueries = 10;
        // const int64_t processed_nq = std::min<int64_t>(run_nq, kMaxQueries);
        const int64_t processed_nq = run_nq;

        std::cout << "Running HNSW search on " << processed_nq
                  << " queries (k=" << k << ", ef=" << sp.ef_search
                  << ", metric=" << (metric == MetricType::L2 ? "L2" : "IP") << ")\n";

        double sum_recall = 0.0;
        int exact_eq = 0;
        int similar_cnt = 0;
        bool do_verify = (faiss_index_ptr != nullptr);

        // you can adjust this threshold as you like:
        const double similar_threshold = 0.90;

        for (int64_t qi = 0; qi < processed_nq; ++qi) {
            const float* qptr = queries.data() + qi * d_q;

            auto topk_custom = hnsw_searchK(G, X, qptr, sp, k);

            if (print_results) {
                std::cout << "q[" << qi << "] custom:";
                for (int t = 0; t < (int)topk_custom.size(); ++t) {
                    std::cout << " (" << topk_custom[t].id << "," << topk_custom[t].d << ")";
                }
            }
            
            // std::cout << "q[" << qi << "] custom:";
            // for (int t = 0; t < (int)topk_custom.size(); ++t) {
            //     std::cout << " (" << topk_custom[t].id << "," << topk_custom[t].d << ")";
            // }

            if (do_verify) {
                auto topk_faiss = faiss_search_topk(faiss_index_ptr.get(), qptr, k, sp.ef_search);
                double r = recall_at_k(topk_custom, topk_faiss);
                bool   eq = exact_match_ids(topk_custom, topk_faiss);
                bool   similar = (r >= similar_threshold);

                sum_recall += r;
                if (eq) ++exact_eq;
                if (similar) ++similar_cnt;
                if (print_results) {
                std::cout << "\n           faiss: ";
                for (int t = 0; t < (int)topk_faiss.size(); ++t) {
                    std::cout << " (" << topk_faiss[t].first << "," << topk_faiss[t].second << ")";
                }
                std::cout << "\n           overlap: " << (int)(r * k) << "/" << k
                          << " recall@k=" << r
                          << (eq ? " [exact@k]" : "")
                          << " similar=" << (similar ? "yes" : "no");
            } 
        }
            if (print_results)
                std::cout << "\n";
        }

        if (do_verify && processed_nq > 0) {
            std::cout << "=== Verification summary (first " << processed_nq << " queries) ===\n";
            std::cout << "avg recall@k: " << (sum_recall / double(processed_nq))
                      << "   exact@k: " << exact_eq << "/" << processed_nq
                      << "   similar(>= " << similar_threshold << "): " << similar_cnt << "/" << processed_nq
                      << "\n";
        }

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 2;
    }
}
