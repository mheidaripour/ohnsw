#pragma once
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

// ---------------------------
// Metrics
// ---------------------------
enum class MetricType { L2, InnerProduct };

inline float l2_sq(const float* a, const float* b, int d) {
    float acc = 0.f;
    for (int i = 0; i < d; ++i) {
        float dx = a[i] - b[i];
        acc += dx * dx;
    }
    return acc;
}

// For IP we *maximize* dot. Convert to a minimization by returning -dot.
inline float neg_dot(const float* a, const float* b, int d) {
    float acc = 0.f;
    for (int i = 0; i < d; ++i) acc += a[i] * b[i];
    return -acc;
}

inline float dist(const float* q, const float* x, int d, MetricType mt) {
    return (mt == MetricType::L2) ? l2_sq(q, x, d) : neg_dot(q, x, d);
}

// ---------------------------
//
// Graph & data views
//
// ---------------------------
struct HNSWView {
    int64_t ntotal = 0;
    int max_level = 0;
    int entry_point = -1;
    // adjacency[layer][node] -> neighbors
    std::vector<std::vector<std::vector<int>>> adjacency;
    // optional: number of levels present for each node (>=1 means present on L0)
    std::vector<int> node_levels;
};

struct MatrixViewF32 {
    const float* data = nullptr;
    int64_t ntotal = 0;
    int d = 0;

    inline const float* row_ptr(int64_t i) const { return data + i * d; }
};

inline bool node_in_layer(const HNSWView& G, int64_t i, int L) {
    if (!G.node_levels.empty()) return G.node_levels[i] > L;
    if (L == 0) return true;
    if (L < 0 || L > G.max_level) return false;
    return !G.adjacency[L][i].empty();
}

// ---------------------------
// Search parameters (FAISS-style knobs)
// ---------------------------
struct SearchParams {
    MetricType metric = MetricType::L2;
    int ef_search = 64;              // FAISS efSearch
    bool bounded_queue = true;       // FAISS default: bounded result heap of size efSearch
    bool check_relative_distance = false; // FAISS flag exists; default off
    float relative_eps = 0.0f;       // epsilon for relative check if enabled
};

// ---------------------------
// Visited table (like FAISS)
// ---------------------------
struct VisitedTable {
    std::vector<int32_t> mark;
    int32_t cur = 1;

    explicit VisitedTable(size_t n = 0) : mark(n, 0) {}
    void reset_if_needed() {
        if (++cur == std::numeric_limits<int32_t>::max()) {
            std::fill(mark.begin(), mark.end(), 0);
            cur = 1;
        }
    }
    void set_size(size_t n) { mark.assign(n, 0); cur = 1; }
    inline bool is_visited(int i) const { return mark[i] == cur; }
    inline void mark_visited(int i)     { mark[i] = cur; }
};

// ---------------------------
// Internal helpers
// ---------------------------
struct Neighbor { int id; float d; };

// min-heap by distance (closest first)
struct CandCmp {
    bool operator()(const Neighbor& a, const Neighbor& b) const { return a.d > b.d; }
};
// max-heap by distance (worst kept on top)
struct ResCmp {
    bool operator()(const Neighbor& a, const Neighbor& b) const { return a.d < b.d; }
};

// Greedy descent on a single upper layer (FAISS behavior).
// Note: FAISS uses strict improvement; we mirror that with (d_nb < d_best).
static int greedy_descent_layer(
    const HNSWView& G,
    const MatrixViewF32& X,
    const float* q,
    MetricType metric,
    int layer,
    int ep)
{
    float d_best = dist(q, X.row_ptr(ep), X.d, metric);
    bool improved = true;
    while (improved) {
        improved = false;
        const auto& nbrs = G.adjacency[layer][ep];
        for (int nb : nbrs) {
            if (nb < 0) continue;
            float d_nb = dist(q, X.row_ptr(nb), X.d, metric);
            if (d_nb < d_best) {
                d_best = d_nb;
                ep = nb;
                improved = true;
            }
        }
    }
    return ep;
}

// Base layer best-first search (FAISS style).
static void search_layer0_faiss(
    const HNSWView& G,
    const MatrixViewF32& X,
    const float* q,
    const SearchParams& sp,
    int ep,
    std::vector<Neighbor>& out)   // unsorted results
{
    out.clear();
    out.reserve(std::max(sp.ef_search, 1));

    std::priority_queue<Neighbor, std::vector<Neighbor>, CandCmp> candidates; // min-heap
    std::priority_queue<Neighbor, std::vector<Neighbor>, ResCmp>   top;       // max-heap

    // Reusable visited table
    static thread_local VisitedTable vt;
    if ((int64_t)vt.mark.size() != G.ntotal) vt.set_size(G.ntotal);
    vt.reset_if_needed();

    // init with entry point
    float d_ep = dist(q, X.row_ptr(ep), X.d, sp.metric);
    candidates.push({ep, d_ep});
    top.push({ep, d_ep});
    vt.mark_visited(ep);

    float d_max = d_ep;

    while (!candidates.empty()) {
        const Neighbor c = candidates.top();

        // FAISS stop condition:
        // - standard: if best candidate is worse than worst in top, break
        // - optional relative check: c.d > d_max * (1 + relative_eps)
        bool stop = false;
        if (sp.check_relative_distance) {
            stop = c.d > d_max * (1.0f + sp.relative_eps);
        } else {
            stop = c.d > d_max;
        }
        if (stop) break;

        candidates.pop();

        const auto& nbrs = G.adjacency[0][c.id];
        for (int nb : nbrs) {
            if (nb < 0) continue;
            if (!vt.is_visited(nb)) {
                vt.mark_visited(nb);

                float d_nb = dist(q, X.row_ptr(nb), X.d, sp.metric);

                // Same insertion rule as FAISS:
                // insert if result set not full OR strictly better than current worst.
                if ((int)top.size() < sp.ef_search || d_nb < top.top().d) {
                    candidates.push({nb, d_nb});
                    top.push({nb, d_nb});
                    if (sp.bounded_queue && (int)top.size() > sp.ef_search) {
                        top.pop(); // keep bounded
                    }
                    d_max = top.top().d;
                }
            }
        }
    }

    // Move heap to vector (unsorted)
    out.reserve(top.size());
    while (!top.empty()) {
        out.push_back(top.top());
        top.pop();
    }
}

// ---------------------------
// Public API: HNSW searchK (FAISS-like)
// ---------------------------
static std::vector<Neighbor> hnsw_searchK(
    const HNSWView& G,
    const MatrixViewF32& X,
    const float* q,
    const SearchParams& sp_in,
    int k)
{
    assert(G.ntotal > 0);
    assert(G.entry_point >= 0 && G.entry_point < G.ntotal);
    assert(!G.adjacency.empty() && (int)G.adjacency.size() == G.max_level + 1);
    assert(X.data && X.ntotal == G.ntotal);

    // Enforce ef >= k (FAISS convention)
    SearchParams sp = sp_in;
    if (sp.ef_search < k) sp.ef_search = k;

    // 1) Greedy descent from top to layer 1
    int ep = G.entry_point;
    for (int L = G.max_level; L > 0; --L) {
        if (node_in_layer(G, ep, L)) {
            ep = greedy_descent_layer(G, X, q, sp.metric, L, ep);
        }
        // If ep not present at L, FAISS entry point is usually the top node; we just skip.
    }

    // 2) Best-first on base layer (L0) with efSearch, bounded queue by default
    std::vector<Neighbor> pool;
    search_layer0_faiss(G, X, q, sp, ep, pool);

    // 3) Return top-k sorted asc
    std::sort(pool.begin(), pool.end(), [](const Neighbor& a, const Neighbor& b){ return a.d < b.d; });
    if ((int)pool.size() > k) pool.resize(k);
    return pool;
}
