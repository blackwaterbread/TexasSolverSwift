#include "cfr_solver.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <cstdio>

namespace texgpu {

// ---------------- kernels ----------------
// All trainable/leaf kernels carry a batch dimension B (the chance-deal fanout).
// Above/at the chance node B=1; below it B=ndeals. Thread t decomposes into
// (b = t / width, h = t % width); per-deal data is laid out [b * stride + ...],
// which matches the persistent trainable layout [deal * nact * nc + ...] so the
// batch index IS the deal/set slot. This makes one batched tree-walk numerically
// identical to the old deal-by-deal recursion but with ndeals-fewer kernel launches.

__device__ __forceinline__ bool disjoint2(int a1, int a2, int b1, int b2) {
    return a1 != b1 && a1 != b2 && a2 != b1 && a2 != b2;
}

// Showdown payoff for `player`. rank table stride = pn (deal b -> prk[b*pn+i]);
// for B==1 (no chance) prk/ork point at base ranks and b is always 0.
// Grid: x tiles player hands, y = batch b. Each block streams the opponent range
// through shared memory once and reuses it across the block's player hands — the
// O(n^2) loop is unavoidable here, but this cuts the repeated global reads that
// dominated it (measured ~58% of a large-range river iteration). Shared layout:
// [oc1 | oc2 | ork] ints then [reach] floats, each blockDim wide.
__global__ void g_showdown_b(const int* pc1, const int* pc2, const int* prk, int pn,
                             const int* oc1, const int* oc2, const int* ork, const float* reach, int on,
                             float win, float lose, float* out, int B) {
    int b = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // player hand in batch b
    const float* rb = reach + (size_t)b * on;
    const int* orkb = ork + (size_t)b * on;

    bool active = (i < pn);
    int ri = active ? prk[(size_t)b * pn + i] : -1;
    int a1 = active ? pc1[i] : -1, a2 = active ? pc2[i] : -1;
    bool valid = active && ri >= 0;
    float acc = 0.0f;

    extern __shared__ int sh[];
    int* s_oc1 = sh;
    int* s_oc2 = sh + blockDim.x;
    int* s_ork = sh + 2 * blockDim.x;
    float* s_reach = (float*)(sh + 3 * blockDim.x);

    for (int base = 0; base < on; base += blockDim.x) {
        int j = base + threadIdx.x;
        if (j < on) {
            s_oc1[threadIdx.x] = oc1[j];
            s_oc2[threadIdx.x] = oc2[j];
            s_ork[threadIdx.x] = orkb[j];
            s_reach[threadIdx.x] = rb[j];
        }
        __syncthreads();
        int tile = min(blockDim.x, on - base);
        if (valid) {
            for (int t = 0; t < tile; ++t) {
                int rj = s_ork[t];
                if (rj < 0) continue;
                int o1 = s_oc1[t], o2 = s_oc2[t];
                if (a1 == o1 || a1 == o2 || a2 == o1 || a2 == o2) continue;  // not disjoint
                float r = s_reach[t];
                if (ri < rj) acc += win * r;
                else if (ri > rj) acc += lose * r;
            }
        }
        __syncthreads();
    }
    if (active) out[(size_t)b * pn + i] = valid ? acc : 0.0f;
}

// ---- O(n) showdown (card-sum trick), no-chance case B==1 ----
// Mirrors PCfrSolver::showdownUtility: player util = win*(weaker-oppo reach) +
// lose*(stronger-oppo reach), excluding card-conflicting oppo hands. The bulk
// (Wtot/Ltot) comes from a rank-sorted prefix sum + binary search; the small
// card-conflict correction loops only the (<=51) oppo hands sharing each player
// card. Replaces the O(n^2) inner loop with O(log n + cards) per player hand.
__device__ __forceinline__ int lb_int(const int* a, int n, int key) {  // first idx with a>=key
    int lo = 0, hi = n;
    while (lo < hi) { int m = (lo + hi) >> 1; if (a[m] < key) lo = m + 1; else hi = m; }
    return lo;
}
__device__ __forceinline__ int ub_int(const int* a, int n, int key) {  // first idx with a>key
    int lo = 0, hi = n;
    while (lo < hi) { int m = (lo + hi) >> 1; if (a[m] <= key) lo = m + 1; else hi = m; }
    return lo;
}

// Exclusive prefix sum of opponent reach in rank-sorted order. pref[k] = reach of
// the k strongest oppo hands; pref[on] = total. Work-efficient (Blelloch) single-
// block scan: capacity n2 = next pow2 >= on, blockDim = n2/2, 2 elements/thread,
// padded with zeros. Holdem on <= 1326 => n2 <= 2048 fits one block.
__global__ void g_sd_scan(const float* reach, const int* order, float* pref, int on, int n2) {
    extern __shared__ float temp[];   // n2 floats
    int tid = threadIdx.x;
    int ai = tid, bi = tid + n2 / 2;
    temp[ai] = (ai < on) ? reach[order[ai]] : 0.0f;
    temp[bi] = (bi < on) ? reach[order[bi]] : 0.0f;

    int offset = 1;
    for (int d = n2 >> 1; d > 0; d >>= 1) {       // up-sweep (reduce)
        __syncthreads();
        if (tid < d) {
            int x = offset * (2 * tid + 1) - 1;
            int y = offset * (2 * tid + 2) - 1;
            temp[y] += temp[x];
        }
        offset <<= 1;
    }
    if (tid == 0) temp[n2 - 1] = 0.0f;
    for (int d = 1; d < n2; d <<= 1) {            // down-sweep
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int x = offset * (2 * tid + 1) - 1;
            int y = offset * (2 * tid + 2) - 1;
            float t = temp[x];
            temp[x] = temp[y];
            temp[y] += t;
        }
    }
    __syncthreads();
    if (ai < on) pref[ai] = temp[ai];
    if (bi < on) pref[bi] = temp[bi];
    // total = exclusive prefix at the last element + that element (avoids temp[on] OOB)
    if (ai == on - 1) pref[on] = temp[ai] + reach[order[ai]];
    if (bi == on - 1) pref[on] = temp[bi] + reach[order[bi]];
}

__global__ void g_sd_eval(const int* prk, int pn, const float* pref, const int* sortedRanks, int on,
                          const int* pc1, const int* pc2, const int* cardOff, const int* cardIdx,
                          const int* orank, const float* reach, float win, float lose, float* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pn) return;
    int ri = prk[i];
    if (ri < 0) { out[i] = 0.0f; return; }
    int lo = lb_int(sortedRanks, on, ri);      // #oppo with rank < ri
    int up = ub_int(sortedRanks, on, ri);      // #oppo with rank <= ri
    float Ltot = pref[lo];                      // stronger oppo (player loses)
    float Wtot = pref[on] - pref[up];           // weaker oppo (player wins)
    int a1 = pc1[i], a2 = pc2[i];
    float Wc = 0.0f, Lc = 0.0f;                 // reach of card-conflicting oppo to exclude
    for (int m = cardOff[a1]; m < cardOff[a1 + 1]; ++m) {
        int j = cardIdx[m]; int rj = orank[j]; float r = reach[j];
        if (rj > ri) Wc += r; else if (rj < ri) Lc += r;
    }
    for (int m = cardOff[a2]; m < cardOff[a2 + 1]; ++m) {
        int j = cardIdx[m]; int rj = orank[j]; float r = reach[j];
        if (rj > ri) Wc += r; else if (rj < ri) Lc += r;
    }
    out[i] = win * (Wtot - Wc) + lose * (Ltot - Lc);
}

// Batched O(n) showdown for chance subgames (B deals). Same card-sum trick, one
// batch per blockIdx.y, using per-deal rank-sorted order/ranks (built host-side
// from dealrank). Invalid hands (rank -1, runout collision) carry zero reach so
// they fall harmlessly into the prefix sums. The per-card CSR is batch-independent.
__global__ void g_sd_scan_b(const float* reach, const int* order, float* pref, int on, int n2) {
    int b = blockIdx.y;
    const float* rb = reach + (size_t)b * on;
    const int* ob = order + (size_t)b * on;
    float* pb = pref + (size_t)b * (on + 1);
    extern __shared__ float temp[];
    int tid = threadIdx.x;
    int ai = tid, bi = tid + n2 / 2;
    temp[ai] = (ai < on) ? rb[ob[ai]] : 0.0f;
    temp[bi] = (bi < on) ? rb[ob[bi]] : 0.0f;
    int offset = 1;
    for (int d = n2 >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) { int x = offset * (2 * tid + 1) - 1, y = offset * (2 * tid + 2) - 1; temp[y] += temp[x]; }
        offset <<= 1;
    }
    if (tid == 0) temp[n2 - 1] = 0.0f;
    for (int d = 1; d < n2; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) { int x = offset * (2 * tid + 1) - 1, y = offset * (2 * tid + 2) - 1; float t = temp[x]; temp[x] = temp[y]; temp[y] += t; }
    }
    __syncthreads();
    if (ai < on) pb[ai] = temp[ai];
    if (bi < on) pb[bi] = temp[bi];
    if (ai == on - 1) pb[on] = temp[ai] + rb[ob[ai]];
    if (bi == on - 1) pb[on] = temp[bi] + rb[ob[bi]];
}

__global__ void g_sd_eval_b(const int* prk, int pn, const float* pref, const int* sortedRanks, int on,
                            const int* pc1, const int* pc2, const int* cardOff, const int* cardIdx,
                            const int* orank, const float* reach, float win, float lose, float* out) {
    int b = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pn) return;
    int ri = prk[(size_t)b * pn + i];
    if (ri < 0) { out[(size_t)b * pn + i] = 0.0f; return; }
    const float* pb = pref + (size_t)b * (on + 1);
    const int* srb = sortedRanks + (size_t)b * on;
    int lo = lb_int(srb, on, ri);
    int up = ub_int(srb, on, ri);
    float Ltot = pb[lo];
    float Wtot = pb[on] - pb[up];
    int a1 = pc1[i], a2 = pc2[i];
    const int* orb = orank + (size_t)b * on;
    const float* rb = reach + (size_t)b * on;
    float Wc = 0.0f, Lc = 0.0f;
    for (int m = cardOff[a1]; m < cardOff[a1 + 1]; ++m) {
        int j = cardIdx[m]; int rj = orb[j]; if (rj < 0) continue; float r = rb[j];
        if (rj > ri) Wc += r; else if (rj < ri) Lc += r;
    }
    for (int m = cardOff[a2]; m < cardOff[a2 + 1]; ++m) {
        int j = cardIdx[m]; int rj = orb[j]; if (rj < 0) continue; float r = rb[j];
        if (rj > ri) Wc += r; else if (rj < ri) Lc += r;
    }
    out[(size_t)b * pn + i] = win * (Wtot - Wc) + lose * (Ltot - Lc);
}

// Terminal (fold). When level>0, batch b is a base-ND compound deal; peel its
// `level` digits and exclude player hands colliding with any dealt card.
__global__ void g_terminal_b(const int* pc1, const int* pc2, int pn,
                             const int* oc1, const int* oc2, const float* reach, int on,
                             float payoff, const int* deal_cards, int ND, int level, float* out, int B) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * pn) return;
    int b = t / pn, i = t % pn;
    int a1 = pc1[i], a2 = pc2[i];
    if (deal_cards && level > 0) {
        int x = b;
        for (int k = 0; k < level; ++k) {
            int dc = deal_cards[x % ND];
            if (a1 == dc || a2 == dc) { out[t] = 0.0f; return; }
            x /= ND;
        }
    }
    const float* rb = reach + (size_t)b * on;
    float acc = 0.0f;
    for (int j = 0; j < on; ++j)
        if (disjoint2(a1, a2, oc1[j], oc2[j])) acc += rb[j];
    out[t] = payoff * acc;
}

// Chance expand: reach[Bin*on] -> out[Bin*ND*on], producing one extra deal level.
// New compound index b_out = b_in*ND + r (output index t == b_out*on + h), so the
// write is contiguous. Zero a slot when card r repeats a card already dealt along
// b_in's path (impossible runout) or collides with the opponent hand; else scale
// the parent reach by 1/possible_deals for this level.
__global__ void g_chance_expand(const float* reach, float* out, const int* oc1, const int* oc2,
                                const int* deal_cards, int ND, int in_level, int on, int Bin, float inv) {
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)Bin * ND * on) return;
    int h = t % on;
    size_t rem = t / on;
    int r = rem % ND;
    int b_in = (int)(rem / ND);
    int card_r = deal_cards[r];
    bool bad = (oc1[h] == card_r || oc2[h] == card_r);
    if (!bad) {
        int x = b_in;
        for (int k = 0; k < in_level; ++k) { if (deal_cards[x % ND] == card_r) { bad = true; break; } x /= ND; }
    }
    out[t] = bad ? 0.0f : reach[(size_t)b_in * on + h] * inv;
}

// Chance reduce: out[b_in*pn + i] = sum over the ND child deals of util.
__global__ void g_chance_reduce(const float* util, float* out, int pn, int ND, int Bin) {
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)Bin * pn) return;
    int i = t % pn, b_in = (int)(t / pn);
    float s = 0.0f;
    for (int r = 0; r < ND; ++r) s += util[((size_t)b_in * ND + r) * pn + i];
    out[t] = s;
}

// rplus/cum are stored as fp16 (half the memory) but all math is done in fp32,
// mirroring the CPU DiscountedCfrTrainableHF mode (storage half, compute float).
__global__ void g_curr_strat_b(const __half* rplus, float* strat, int nact, int nc, int B) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * nc) return;
    int b = t / nc, h = t % nc;
    const __half* rp = rplus + (size_t)b * nact * nc;
    float* st = strat + (size_t)b * nact * nc;
    float s = 0.0f;
    for (int a = 0; a < nact; ++a) { float v = __half2float(rp[a * nc + h]); if (v > 0.0f) s += v; }
    for (int a = 0; a < nact; ++a) {
        int idx = a * nc + h;
        float v = __half2float(rp[idx]);
        st[idx] = (s > 0.0f) ? (v > 0.0f ? v / s : 0.0f) : (1.0f / nact);
    }
}

__global__ void g_row_mul_b(const float* reach, const float* strat, float* out, int a, int nc, int nact, int B) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * nc) return;
    int b = t / nc, h = t % nc;
    out[t] = reach[t] * strat[(size_t)b * nact * nc + a * nc + h];
}

__global__ void g_add(float* dst, const float* src, int n) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h < n) dst[h] += src[h];
}

// Best-response helpers (exploitability diagnostic): fill and elementwise max.
__global__ void g_fill(float* d, float v, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = v;
}
__global__ void g_max_b(float* dst, const float* src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = fmaxf(dst[i], src[i]);
}

// dst[b*nc+h] += strat[b][a][h] * util[b*nc+h]   (pn == nc for own-action nodes)
__global__ void g_fma_strat_b(float* dst, const float* strat, const float* util, int a, int nc, int nact, int B) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * nc) return;
    int b = t / nc, h = t % nc;
    dst[t] += strat[(size_t)b * nact * nc + a * nc + h] * util[t];
}

__global__ void g_set_regret_row_b(float* regret, const float* util, const float* pay, int a, int nc, int nact, int B) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * nc) return;
    int b = t / nc, h = t % nc;
    regret[(size_t)b * nact * nc + a * nc + h] = util[t] - pay[t];
}

// DCFR discount coefficients for this iteration, kept device-resident so the
// captured CUDA graph stays static across iterations (only this 2-float buffer
// changes per replay). alpha=1.5, gamma=2 (beta/theta are constants in g_update_b).
__global__ void g_set_coefs(float* coefs, int iter) {
    float t = (float)(iter + 1);
    float a = powf(t, 1.5f);
    coefs[0] = a / (1.0f + a);                  // alpha_coef
    coefs[1] = powf(t / (t + 1.0f), 2.0f);      // strat_coef
}

__global__ void g_update_b(const float* regret, __half* rplus, __half* cum, int nact, int nc,
                           const float* coefs, float beta, float theta, int B) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * nc) return;
    int b = t / nc, h = t % nc;
    float alpha_coef = coefs[0], strat_coef = coefs[1];
    const float* rg = regret + (size_t)b * nact * nc;
    __half* rp = rplus + (size_t)b * nact * nc;
    __half* cm = cum + (size_t)b * nact * nc;
    float rsum = 0.0f;
    for (int a = 0; a < nact; ++a) {
        int idx = a * nc + h;
        float v = rg[idx] + __half2float(rp[idx]);
        v = (v > 0.0f) ? v * alpha_coef : v * beta;
        rp[idx] = __float2half(v);
        if (v > 0.0f) rsum += v;
    }
    for (int a = 0; a < nact; ++a) {
        int idx = a * nc + h;
        float rpv = __half2float(rp[idx]);
        float s = (rsum > 0.0f) ? (rpv > 0.0f ? rpv / rsum : 0.0f) : (1.0f / nact);
        cm[idx] = __float2half(__half2float(cm[idx]) * theta + s * strat_coef);
    }
}

__global__ void g_avg(const __half* cum, float* avg, int nact, int nc) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= nc) return;
    float c = 0.0f;
    for (int a = 0; a < nact; ++a) c += __half2float(cum[a * nc + h]);
    for (int a = 0; a < nact; ++a) {
        int idx = a * nc + h;
        avg[idx] = (c > 0.0f) ? __half2float(cum[idx]) / c : (1.0f / nact);
    }
}

static inline int blocks_for(size_t n, int t) { return (int)((n + t - 1) / t); }

// Use the O(n) card-sum showdown when the opponent range is at least this large;
// below it the O(n^2) tiled kernel avoids the per-batch scan overhead (measured:
// tiled wins for tiny nc like the flop test's 13, O(n) wins by nc~37+).
static const int kShowdownOnThreshold = 32;

// ---------------- solver ----------------

static inline int ipow(int base, int e) { int r = 1; for (int i = 0; i < e; ++i) r *= base; return r; }

// Chance depth of a node = streets dealt above it = node.round - root_round.
int CudaCfrSolver::level(int nodeid) const {
    return sg_.nodes[nodeid].round - root_round_;
}

// Trainable sets at an action node = ND^level (one per compound runout reaching it).
int CudaCfrSolver::ntrainsets(int nodeid) const {
    return ipow(ND_, level(nodeid));
}

CudaCfrSolver::CudaCfrSolver(const Subgame& sg) : sg_(sg) {
    root_round_ = sg.nodes[sg.root].round;
    ND_ = std::max(1, sg.ndeals());

    for (int p = 0; p < 2; ++p) {
        int n = sg.ncombos(p);
        ncards_[p] = n;
        std::vector<int> c1(n), c2(n), rk(n);
        for (int i = 0; i < n; ++i) { c1[i] = sg.ranges[p][i].card1; c2[i] = sg.ranges[p][i].card2; rk[i] = sg.ranges[p][i].rank; }
        cudaMalloc(&d_c1_[p], n * sizeof(int)); cudaMemcpy(d_c1_[p], c1.data(), n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMalloc(&d_c2_[p], n * sizeof(int)); cudaMemcpy(d_c2_[p], c2.data(), n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMalloc(&d_rank_[p], n * sizeof(int)); cudaMemcpy(d_rank_[p], rk.data(), n * sizeof(int), cudaMemcpyHostToDevice);
        if (sg.has_chance && !sg.dealrank[p].empty()) {
            size_t sz = sg.dealrank[p].size();
            cudaMalloc(&d_dealrank_[p], sz * sizeof(int));
            cudaMemcpy(d_dealrank_[p], sg.dealrank[p].data(), sz * sizeof(int), cudaMemcpyHostToDevice);
        }
    }

    if (sg.has_chance && !sg.deal_cards.empty()) {
        size_t sz = sg.deal_cards.size();
        cudaMalloc(&d_deal_cards_, sz * sizeof(int));
        cudaMemcpy(d_deal_cards_, sg.deal_cards.data(), sz * sizeof(int), cudaMemcpyHostToDevice);
    }

    // O(n) showdown structures (used only for the no-chance river case, B==1):
    // rank-sorted order + per-card CSR over base ranks, built once per range side.
    for (int O = 0; O < 2; ++O) {
        int n = ncards_[O];
        const std::vector<Combo>& rg = sg.ranges[O];
        std::vector<int> order(n);
        for (int i = 0; i < n; ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](int x, int y) { return rg[x].rank < rg[y].rank; });
        std::vector<int> sorted_ranks(n);
        for (int k = 0; k < n; ++k) sorted_ranks[k] = rg[order[k]].rank;
        std::vector<std::vector<int>> byCard(52);
        for (int j = 0; j < n; ++j) { byCard[rg[j].card1].push_back(j); byCard[rg[j].card2].push_back(j); }
        std::vector<int> cardoff(53, 0), cardidx;
        cardidx.reserve(2 * n);
        for (int c = 0; c < 52; ++c) { cardoff[c + 1] = cardoff[c] + (int)byCard[c].size(); for (int j : byCard[c]) cardidx.push_back(j); }
        cudaMalloc(&d_rankorder_[O], n * sizeof(int));   cudaMemcpy(d_rankorder_[O], order.data(), n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMalloc(&d_sortedranks_[O], n * sizeof(int)); cudaMemcpy(d_sortedranks_[O], sorted_ranks.data(), n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMalloc(&d_cardoff_[O], 53 * sizeof(int));    cudaMemcpy(d_cardoff_[O], cardoff.data(), 53 * sizeof(int), cudaMemcpyHostToDevice);
        cudaMalloc(&d_cardidx_[O], cardidx.size() * sizeof(int)); cudaMemcpy(d_cardidx_[O], cardidx.data(), cardidx.size() * sizeof(int), cudaMemcpyHostToDevice);
    }

    // Per-deal rank-sorted order + sorted ranks for the batched O(n) showdown in
    // chance subgames (built from the loaded dealrank; card CSR above is reused).
    if (sg.has_chance) {
        for (int O = 0; O < 2; ++O) {
            int nc = ncards_[O];
            if (nc == 0 || sg.dealrank[O].empty()) continue;
            long long nrows = (long long)sg.dealrank[O].size() / nc;
            std::vector<int> order(sg.dealrank[O].size()), sranks(sg.dealrank[O].size());
            std::vector<int> idx(nc);
            for (long long b = 0; b < nrows; ++b) {
                const int* row = sg.dealrank[O].data() + b * nc;
                for (int k = 0; k < nc; ++k) idx[k] = k;
                std::sort(idx.begin(), idx.end(), [&](int x, int y) { return row[x] < row[y]; });
                for (int k = 0; k < nc; ++k) { order[b * nc + k] = idx[k]; sranks[b * nc + k] = row[idx[k]]; }
            }
            cudaMalloc(&d_dealorder_[O], order.size() * sizeof(int));
            cudaMemcpy(d_dealorder_[O], order.data(), order.size() * sizeof(int), cudaMemcpyHostToDevice);
            cudaMalloc(&d_dealsortedranks_[O], sranks.size() * sizeof(int));
            cudaMemcpy(d_dealsortedranks_[O], sranks.data(), sranks.size() * sizeof(int), cudaMemcpyHostToDevice);
        }
    }

    int N = (int)sg.nodes.size();
    d_rplus_.assign(N, nullptr);
    d_cum_.assign(N, nullptr);
    nact_.assign(N, 0);
    for (int i = 0; i < N; ++i) {
        const Node& nd = sg.nodes[i];
        if (nd.type != NT_ACTION) continue;
        int nact = (int)nd.children.size();
        int nc = ncards_[nd.player];
        int sets = ntrainsets(i);
        size_t sz = (size_t)sets * nact * nc;
        nact_[i] = nact;
        cudaMalloc(&d_rplus_[i], sz * sizeof(__half)); cudaMemset(d_rplus_[i], 0, sz * sizeof(__half));
        cudaMalloc(&d_cum_[i], sz * sizeof(__half)); cudaMemset(d_cum_[i], 0, sz * sizeof(__half));
    }

    // LIFO scratch arena. Batched walk widens every scratch buffer by the node's
    // B = ND^level; the deepest level reaches ND^chance_levels. Peak usage is one
    // root-to-leaf path, tiny vs this floor for the modest test ranges.
    int maxnc = std::max(ncards_[0], ncards_[1]);
    int maxB = ipow(ND_, sg.chance_levels);
    size_t need = (size_t)maxnc * 64 * (size_t)maxB;
    arena_cap_ = std::max((size_t)256 * 1024 * 1024 / sizeof(float), need);
    cudaMalloc(&arena_, arena_cap_ * sizeof(float));

    cudaStreamCreate(&stream_);
    cudaMalloc(&d_coefs_, 2 * sizeof(float));
}

CudaCfrSolver::~CudaCfrSolver() {
    for (int p = 0; p < 2; ++p) {
        cudaFree(d_c1_[p]); cudaFree(d_c2_[p]); cudaFree(d_rank_[p]);
        if (d_dealrank_[p]) cudaFree(d_dealrank_[p]);
    }
    if (d_deal_cards_) cudaFree(d_deal_cards_);
    for (int O = 0; O < 2; ++O) {
        cudaFree(d_rankorder_[O]); cudaFree(d_sortedranks_[O]);
        cudaFree(d_cardoff_[O]); cudaFree(d_cardidx_[O]);
        if (d_dealorder_[O]) cudaFree(d_dealorder_[O]);
        if (d_dealsortedranks_[O]) cudaFree(d_dealsortedranks_[O]);
    }
    for (auto p : d_rplus_) cudaFree(p);
    for (auto p : d_cum_) cudaFree(p);
    cudaFree(arena_);
    if (graph_exec_) cudaGraphExecDestroy(graph_exec_);
    if (d_coefs_) cudaFree(d_coefs_);
    cudaStreamDestroy(stream_);
}

float* CudaCfrSolver::arena_alloc(size_t n) {
    size_t a = (arena_top_ + 31) & ~size_t(31);
    if (a + n > arena_cap_) { printf("FATAL: device arena overflow\n"); abort(); }
    arena_top_ = a + n;
    return arena_ + a;
}

// d_reach: [B*on], d_out: [B*pn]. B = ndeals below the chance node, else 1.
// All work is issued on stream_ so train() can capture one walk as a CUDA graph.
// br=true computes a best-response value instead of training: the cfr-player maxes
// over actions and the opponent plays the precomputed average strategy (d_avgstrat_);
// no regret/strategy updates. Used post-training for the exploitability metric.
void CudaCfrSolver::cfr(int player, int nodeid, const float* d_reach, float* d_out, bool br) {
    const Node& nd = sg_.nodes[nodeid];
    const int T = 128;
    int pn = ncards_[player];                 // util indexed by cfr player's range
    int on = ncards_[1 - player];             // reach indexed by opponent's range
    int lvl = level(nodeid);                  // chance depth (0 above all chances)
    int B = ipow(ND_, lvl);                   // batch = compound runouts reaching here
    bool use_dealrank = sg_.has_chance && lvl >= 1;

    if (nd.type == NT_SHOWDOWN) {
        float win = (float)nd.sd[player][player];
        float lose = (float)nd.sd[1 - player][player];
        if (B == 1) {
            // O(n) card-sum trick (no chance => base ranks, single batch).
            int O = 1 - player;
            float* d_pref = arena_alloc((size_t)on + 1);
            int n2 = 2; while (n2 < on) n2 <<= 1;        // capacity (pow2 >= on)
            g_sd_scan<<<1, n2 / 2, n2 * sizeof(float), stream_>>>(d_reach, d_rankorder_[O], d_pref, on, n2);
            g_sd_eval<<<blocks_for((size_t)pn, T), T, 0, stream_>>>(
                d_rank_[player], pn, d_pref, d_sortedranks_[O], on,
                d_c1_[player], d_c2_[player], d_cardoff_[O], d_cardidx_[O],
                d_rank_[O], d_reach, win, lose, d_out);
            return;
        }
        // Chance subgame (B deals). For larger ranges the O(n) card-sum (batched
        // over deals) beats the O(n^2) tiled kernel; for tiny ranges the tiled
        // kernel avoids the per-batch scan overhead.
        if (on >= kShowdownOnThreshold) {
            int O = 1 - player;
            float* d_pref = arena_alloc((size_t)B * ((size_t)on + 1));
            int n2 = 2; while (n2 < on) n2 <<= 1;
            dim3 scan_grid(1, B);
            g_sd_scan_b<<<scan_grid, n2 / 2, n2 * sizeof(float), stream_>>>(d_reach, d_dealorder_[O], d_pref, on, n2);
            dim3 eval_grid(blocks_for((size_t)pn, T), B);
            g_sd_eval_b<<<eval_grid, T, 0, stream_>>>(d_dealrank_[player], pn, d_pref, d_dealsortedranks_[O], on,
                d_c1_[player], d_c2_[player], d_cardoff_[O], d_cardidx_[O], d_dealrank_[1 - player], d_reach, win, lose, d_out);
            return;
        }
        const int* prk = use_dealrank ? d_dealrank_[player] : d_rank_[player];
        const int* ork = use_dealrank ? d_dealrank_[1 - player] : d_rank_[1 - player];
        dim3 sd_grid(blocks_for((size_t)pn, T), B);
        size_t sd_shmem = (size_t)T * (3 * sizeof(int) + sizeof(float));
        g_showdown_b<<<sd_grid, T, sd_shmem, stream_>>>(d_c1_[player], d_c2_[player], prk, pn,
                                                        d_c1_[1 - player], d_c2_[1 - player], ork, d_reach, on,
                                                        win, lose, d_out, B);
        return;
    }
    if (nd.type == NT_TERMINAL) {
        float payoff = (float)nd.pay[player];
        const int* dc = use_dealrank ? d_deal_cards_ : nullptr;
        g_terminal_b<<<blocks_for((size_t)B * pn, T), T, 0, stream_>>>(d_c1_[player], d_c2_[player], pn,
                                                           d_c1_[1 - player], d_c2_[1 - player], d_reach, on,
                                                           payoff, dc, ND_, lvl, d_out, B);
        return;
    }
    if (nd.type == NT_CHANCE) {
        int child = nd.children[0];
        int out_level = lvl;                  // chance node's round = the round it deals into
        int in_level = out_level - 1;
        int Bin = ipow(ND_, in_level);        // batch entering the chance
        float inv = 1.0f / (float)sg_.possible_deals[out_level - 1];
        size_t mark = arena_top_;
        float* d_nr = arena_alloc((size_t)Bin * ND_ * on);
        g_chance_expand<<<blocks_for((size_t)Bin * ND_ * on, T), T, 0, stream_>>>(d_reach, d_nr,
                        d_c1_[1 - player], d_c2_[1 - player], d_deal_cards_, ND_, in_level, on, Bin, inv);
        float* d_cu = arena_alloc((size_t)Bin * ND_ * pn);
        cfr(player, child, d_nr, d_cu, br);       // child one level deeper => B=Bin*ND
        g_chance_reduce<<<blocks_for((size_t)Bin * pn, T), T, 0, stream_>>>(d_cu, d_out, pn, ND_, Bin);
        arena_top_ = mark;
        return;
    }

    // ACTION node
    int np = nd.player;
    int nact = nact_[nodeid];
    int nc = ncards_[np];

    if (br) {
        // Best response: cfr-player maxes over actions; opponent plays avg strategy.
        size_t mark = arena_top_;
        float* utils = arena_alloc((size_t)nact * B * pn);
        if (np == player) {                         // BR player: reach unchanged, max
            for (int a = 0; a < nact; ++a)
                cfr(player, nd.children[a], d_reach, utils + (size_t)a * B * pn, true);
            g_fill<<<blocks_for((size_t)B * pn, T), T, 0, stream_>>>(d_out, -1e30f, (int)((size_t)B * pn));
            for (int a = 0; a < nact; ++a)
                g_max_b<<<blocks_for((size_t)B * pn, T), T, 0, stream_>>>(d_out, utils + (size_t)a * B * pn, (int)((size_t)B * pn));
        } else {                                    // opponent: avg-strategy reach, sum
            float* avg = d_avgstrat_[nodeid];       // [B*nact*nc] average strategy
            for (int a = 0; a < nact; ++a) {
                float* d_newreach = arena_alloc((size_t)B * nc);
                g_row_mul_b<<<blocks_for((size_t)B * nc, T), T, 0, stream_>>>(d_reach, avg, d_newreach, a, nc, nact, B);
                cfr(player, nd.children[a], d_newreach, utils + (size_t)a * B * pn, true);
            }
            cudaMemsetAsync(d_out, 0, (size_t)B * pn * sizeof(float), stream_);
            for (int a = 0; a < nact; ++a)
                g_add<<<blocks_for((size_t)B * pn, T), T, 0, stream_>>>(d_out, utils + (size_t)a * B * pn, (int)((size_t)B * pn));
        }
        arena_top_ = mark;
        return;
    }

    __half* rplus = d_rplus_[nodeid];   // [B*nact*nc] fp16, slot b == deal b
    __half* cum = d_cum_[nodeid];

    size_t mark = arena_top_;
    float* d_strat = arena_alloc((size_t)B * nact * nc);
    g_curr_strat_b<<<blocks_for((size_t)B * nc, T), T, 0, stream_>>>(rplus, d_strat, nact, nc, B);

    float* utils = arena_alloc((size_t)nact * B * pn);   // [a][b][pn]
    for (int a = 0; a < nact; ++a) {
        float* util_a = utils + (size_t)a * B * pn;
        if (np != player) {
            float* d_newreach = arena_alloc((size_t)B * nc);   // nc == on here
            g_row_mul_b<<<blocks_for((size_t)B * nc, T), T, 0, stream_>>>(d_reach, d_strat, d_newreach, a, nc, nact, B);
            cfr(player, nd.children[a], d_newreach, util_a);
        } else {
            cfr(player, nd.children[a], d_reach, util_a);
        }
    }

    cudaMemsetAsync(d_out, 0, (size_t)B * pn * sizeof(float), stream_);
    if (np == player) {
        for (int a = 0; a < nact; ++a)
            g_fma_strat_b<<<blocks_for((size_t)B * nc, T), T, 0, stream_>>>(d_out, d_strat, utils + (size_t)a * B * pn, a, nc, nact, B);
    } else {
        for (int a = 0; a < nact; ++a)
            g_add<<<blocks_for((size_t)B * pn, T), T, 0, stream_>>>(d_out, utils + (size_t)a * B * pn, (int)((size_t)B * pn));
    }

    if (np == player) {
        float* d_regret = arena_alloc((size_t)B * nact * nc);
        for (int a = 0; a < nact; ++a)
            g_set_regret_row_b<<<blocks_for((size_t)B * nc, T), T, 0, stream_>>>(d_regret, utils + (size_t)a * B * pn, d_out, a, nc, nact, B);
        g_update_b<<<blocks_for((size_t)B * nc, T), T, 0, stream_>>>(d_regret, rplus, cum, nact, nc,
                                                         d_coefs_, 0.5f /*beta*/, 0.9f /*theta*/, B);
    }

    arena_top_ = mark;
}

// One CFR iteration = walk the tree for both players. The kernel sequence is
// identical every iteration (same tree, same order, same arena offsets), so we
// capture it once as a CUDA graph and replay it — eliminating the per-launch host
// overhead that dominates these small, launch-bound subgames. The only per-iter
// input is the DCFR coefficient pair, set by g_set_coefs into d_coefs_ before
// each replay (outside the graph, so the graph stays static).
void CudaCfrSolver::runIteration() {
    for (int player = 0; player < 2; ++player) {
        float* d_out = arena_alloc(ncards_[player]);   // root: B=1
        cfr(player, sg_.root, d_init_[1 - player], d_out);
        arena_top_ = 0;
    }
}

double CudaCfrSolver::train(int iters) {
    for (int p = 0; p < 2; ++p) {
        int n = ncards_[p];
        std::vector<float> w(n);
        for (int i = 0; i < n; ++i) w[i] = sg_.ranges[p][i].weight;
        cudaMalloc(&d_init_[p], n * sizeof(float));
        cudaMemcpy(d_init_[p], w.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    }

    // Capture one iteration's kernel stream into a replayable graph.
    cudaGraph_t graph = nullptr;
    cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal);
    runIteration();
    cudaStreamEndCapture(stream_, &graph);
    cudaGraphInstantiate(&graph_exec_, graph, nullptr, nullptr, 0);
    cudaGraphDestroy(graph);

    cudaDeviceSynchronize();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iters; ++it) {
        g_set_coefs<<<1, 1, 0, stream_>>>(d_coefs_, it);   // per-iter, outside the graph
        cudaGraphLaunch(graph_exec_, stream_);
    }
    cudaStreamSynchronize(stream_);
    auto t1 = std::chrono::high_resolution_clock::now();

    cudaFree(d_init_[0]); cudaFree(d_init_[1]);
    d_init_[0] = d_init_[1] = nullptr;
    return std::chrono::duration<double>(t1 - t0).count();
}

std::vector<std::vector<float>> CudaCfrSolver::averageStrategies() {
    const int T = 128;
    int N = (int)sg_.nodes.size();
    std::vector<std::vector<float>> out(N);
    for (int i = 0; i < N; ++i) {
        if (sg_.nodes[i].type != NT_ACTION) continue;
        int nact = nact_[i];
        int nc = ncards_[sg_.nodes[i].player];
        int sets = ntrainsets(i);
        size_t sz = (size_t)sets * nact * nc;
        float* d_avg = nullptr; cudaMalloc(&d_avg, sz * sizeof(float));
        for (int s = 0; s < sets; ++s)
            g_avg<<<blocks_for(nc, T), T>>>(d_cum_[i] + (size_t)s * nact * nc,
                                            d_avg + (size_t)s * nact * nc, nact, nc);
        cudaDeviceSynchronize();
        out[i].resize(sz);
        cudaMemcpy(out[i].data(), d_avg, sz * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_avg);
    }
    return out;
}

double CudaCfrSolver::exploitability() {
    int N = (int)sg_.nodes.size();
    // Upload the average strategy per action node for the best-response traversal.
    std::vector<std::vector<float>> avgs = averageStrategies();
    d_avgstrat_.assign(N, nullptr);
    for (int i = 0; i < N; ++i) {
        if (sg_.nodes[i].type != NT_ACTION) continue;
        size_t sz = avgs[i].size();
        cudaMalloc(&d_avgstrat_[i], sz * sizeof(float));
        cudaMemcpy(d_avgstrat_[i], avgs[i].data(), sz * sizeof(float), cudaMemcpyHostToDevice);
    }

    // Opponent reach init = range weights (same as training).
    float* d_w[2] = {nullptr, nullptr};
    std::vector<float> w[2];
    for (int p = 0; p < 2; ++p) {
        int n = ncards_[p];
        w[p].resize(n);
        for (int i = 0; i < n; ++i) w[p][i] = sg_.ranges[p][i].weight;
        cudaMalloc(&d_w[p], n * sizeof(float));
        cudaMemcpy(d_w[p], w[p].data(), n * sizeof(float), cudaMemcpyHostToDevice);
    }

    // Best-response value per player: opponent plays avg, this player best-responds.
    double brVal[2] = {0, 0};
    for (int P = 0; P < 2; ++P) {
        arena_top_ = 0;
        float* d_out = arena_alloc(ncards_[P]);
        cfr(P, sg_.root, d_w[1 - P], d_out, true);
        cudaStreamSynchronize(stream_);
        std::vector<float> cfv(ncards_[P]);
        cudaMemcpy(cfv.data(), d_out, ncards_[P] * sizeof(float), cudaMemcpyDeviceToHost);
        double s = 0;
        for (int h = 0; h < ncards_[P]; ++h) s += (double)w[P][h] * cfv[h];
        brVal[P] = s;
    }
    arena_top_ = 0;

    // Normalization mass = sum of disjoint matchup reach products.
    double M = 0;
    for (int i = 0; i < ncards_[0]; ++i)
        for (int j = 0; j < ncards_[1]; ++j) {
            const Combo& a = sg_.ranges[0][i];
            const Combo& b = sg_.ranges[1][j];
            if (a.card1 != b.card1 && a.card1 != b.card2 && a.card2 != b.card1 && a.card2 != b.card2)
                M += (double)a.weight * b.weight;
        }

    for (int i = 0; i < N; ++i) if (d_avgstrat_[i]) cudaFree(d_avgstrat_[i]);
    d_avgstrat_.clear();
    cudaFree(d_w[0]); cudaFree(d_w[1]);

    double pot = sg_.nodes[sg_.root].pot;
    double exploit = (M > 0) ? (brVal[0] + brVal[1]) / M : 0.0;   // chips/matchup, ->0 at Nash
    printf("exploitability: BR0=%.4f BR1=%.4f  exploit=%.6f chips (%.4f%% of pot %.2f)\n",
           brVal[0] / M, brVal[1] / M, exploit, pot > 0 ? 100.0 * exploit / pot : 0.0, pot);
    return exploit;
}

} // namespace texgpu
