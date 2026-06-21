#include "cfr_solver.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <cstdio>
#include <cstring>

// Available physical host RAM, used to decide whether host-streaming the (GB-scale)
// river trainable is safe. Pinning/allocating more than fits destabilizes the OS, so
// streaming declines and falls back when this is too small. Returns 0 if unknown.
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
static size_t host_avail_bytes() { MEMORYSTATUSEX s; s.dwLength = sizeof(s); return GlobalMemoryStatusEx(&s) ? (size_t)s.ullAvailPhys : 0; }
#else
#include <unistd.h>
static size_t host_avail_bytes() {
    long pages = sysconf(_SC_AVPHYS_PAGES), ps = sysconf(_SC_PAGESIZE);
    return (pages > 0 && ps > 0) ? (size_t)pages * ps : 0;
}
#endif

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

// Terminal (fold). When level>0, batch b is a mixed-radix compound deal: digit for
// the deepest level (river, radix nd2) is least-significant, the shallower level
// (turn, radix nd1) most-significant. Peel the `level` digits and exclude player
// hands colliding with any dealt card. (At most 2 chance levels in this solver.)
__global__ void g_terminal_b(const int* pc1, const int* pc2, int pn,
                             const int* oc1, const int* oc2, const float* reach, int on,
                             float payoff, const int* lvl1cards, int nd1, const int* lvl2cards, int nd2,
                             int level, float* out, int B,
                             int riv_iso, const int* slot2turn, const int* rivrepcards, int extra_card) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * pn) return;
    int b = t / pn, i = t % pn;
    int a1 = pc1[i], a2 = pc2[i];
    // Streamed river subtree: b is the river digit (level==1 over the river set) and the
    // turn-deal card is supplied separately (it is not encoded in b). Exclude it too.
    if (extra_card >= 0 && (a1 == extra_card || a2 == extra_card)) { out[t] = 0.0f; return; }
    if (riv_iso && level == 2) {
        // Ragged level-2: batch b is an absolute river-rep slot. Exclude hands hitting
        // either the turn-rep card (lvl1cards[slot2turn[b]]) or the river-rep card.
        int tcard = lvl1cards[slot2turn[b]];
        int rcard = rivrepcards[b];
        if (a1 == tcard || a2 == tcard || a1 == rcard || a2 == rcard) { out[t] = 0.0f; return; }
    } else if (level > 0) {
        int x = b;
        for (int L = level; L >= 1; --L) {       // deepest (river) digit first
            int rad = (L == 2) ? nd2 : nd1;
            const int* cards = (L == 2) ? lvl2cards : lvl1cards;
            int dc = cards[x % rad];
            if (a1 == dc || a2 == dc) { out[t] = 0.0f; return; }
            x /= rad;
        }
    }
    const float* rb = reach + (size_t)b * on;
    float acc = 0.0f;
    for (int j = 0; j < on; ++j)
        if (disjoint2(a1, a2, oc1[j], oc2[j])) acc += rb[j];
    out[t] = payoff * acc;
}

// Chance expand: reach[Bin*on] -> out[Bin*nd_cur*on], producing one extra deal
// level. curcards/nd_cur are this level's deal set; b_out = b_in*nd_cur + r (output
// index t == b_out*on + h), so the write is contiguous. Zero a slot when card r
// repeats a card already dealt along b_in's path (impossible runout) or collides
// with the opponent hand; else scale the parent reach by 1/possible_deals. b_in's
// path digits are the shallower levels (only level 1 exists for in_level<=1).
__global__ void g_chance_expand(const float* reach, float* out, const int* oc1, const int* oc2,
                                const int* curcards, int nd_cur, const int* lvl1cards, int nd1,
                                int in_level, int on, int Bin, float inv) {
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)Bin * nd_cur * on) return;
    int h = t % on;
    size_t rem = t / on;
    int r = rem % nd_cur;
    int b_in = (int)(rem / nd_cur);
    int card_r = curcards[r];
    bool bad = (oc1[h] == card_r || oc2[h] == card_r);
    if (!bad) {
        int x = b_in;
        for (int L = in_level; L >= 1; --L) {    // path: shallower dealt cards (level 1)
            if (lvl1cards[x % nd1] == card_r) { bad = true; break; }
            x /= nd1;
        }
    }
    out[t] = bad ? 0.0f : reach[(size_t)b_in * on + h] * inv;
}

// Streamed river chance EXPAND (one turn deal at a time). The parent reach is the
// single turn-deal row [on]; produce one river reach row per river card r ([ND*on],
// index r*on+h). Zero a slot when the river card hits the opponent hand or repeats
// the turn-deal card (turncard); else scale by 1/possible_deals. Equivalent to one
// b_in slice of g_chance_expand but with the turn card passed explicitly (b carries
// only the river digit in the streamed walk).
__global__ void g_chance_expand_stream(const float* reach_row, float* out, const int* oc1, const int* oc2,
                                       const int* rivercards, int ND, int turncard, int on, float inv) {
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)ND * on) return;
    int h = (int)(t % on);
    int r = (int)(t / on);
    int card_r = rivercards[r];
    bool bad = (oc1[h] == card_r || oc2[h] == card_r || card_r == turncard);
    out[t] = bad ? 0.0f : reach_row[h] * inv;
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

// Chance reduce with suit isomorphism. The children hold only the ND_iso
// representative runouts; re-expand to all nd_full real runouts by, for each full
// card c, reading representative slot rep_slot[c] with this player's hands relabeled
// by the suit permutation perm[c*pn + i]. Mirrors the CPU iso scheme (representative
// utility recovered for each equivalent runout via exchange_color, then summed).
__global__ void g_chance_reduce_iso(const float* util, float* out, int pn, int ND_iso, int Bin,
                                    int nd_full, const int* rep_slot, const int* perm) {
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)Bin * pn) return;
    int i = t % pn, b_in = (int)(t / pn);
    float s = 0.0f;
    for (int c = 0; c < nd_full; ++c)
        s += util[((size_t)b_in * ND_iso + rep_slot[c]) * pn + perm[(size_t)c * pn + i]];
    out[t] = s;
}

// Full 2-level iso (flop) level-2 chance EXPAND. The reach entering is one row per
// turn rep (Bin = NT). Output is one row per absolute river-rep slot: slot's owning
// turn rep tr = slot2turn[slot] supplies the parent reach, masked when the river-rep
// card collides with the opponent hand and scaled by 1/possible_deals (the turn-rep
// card never repeats a river rep). Output index == slot*on+h is contiguous because
// slots are ordered by turn rep then river rep (prefix offsets).
__global__ void g_chance_expand_riv2(const float* reach, float* out, const int* oc1, const int* oc2,
                                     const int* slot2turn, const int* rivrepcards, int on, int total, float inv) {
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)total * on) return;
    int h = (int)(t % on);
    int slot = (int)(t / on);
    int tr = slot2turn[slot];
    int card_r = rivrepcards[slot];
    bool bad = (oc1[h] == card_r || oc2[h] == card_r);
    out[t] = bad ? 0.0f : reach[(size_t)tr * on + h] * inv;
}

// Full 2-level iso (flop) level-2 chance REDUCE. Sum each turn rep's full rivers,
// reading the river-rep utility relabeled by the per-(turn rep, full river) hand
// permutation. fullslot[tr*ND + r] is the abs river-rep slot (-1 when river r is the
// turn-rep card). Output is one row per turn rep (Bin = NT), feeding the level-1
// turn reduce. Mirrors the CPU iso scheme one level deeper.
__global__ void g_chance_reduce_riv2(const float* util, float* out, int pn, int NT, int ND,
                                     const int* fullslot, const int* perm) {
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)NT * pn) return;
    int i = (int)(t % pn), tr = (int)(t / pn);
    float s = 0.0f;
    for (int r = 0; r < ND; ++r) {
        int slot = fullslot[(size_t)tr * ND + r];
        if (slot < 0) continue;
        int ph = perm[((size_t)tr * ND + r) * pn + i];
        s += util[(size_t)slot * pn + ph];
    }
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

// Fused per-action kernels (one launch instead of nact). All are algorithm-
// identical to the per-action loops they replace; they cut the captured graph's
// node count and widen each kernel's grid for better occupancy.

// Opponent node: scale the opponent reach by every action's strategy at once.
// out[a*B*nc + b*nc + h] = reach[b*nc+h] * strat[(b*nact+a)*nc + h]
__global__ void g_row_mul_all(const float* reach, const float* strat, float* out, int nc, int nact, int B) {
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)nact * B * nc;
    if (t >= total) return;
    size_t bn = (size_t)B * nc;
    int a = (int)(t / bn);
    size_t r = t - (size_t)a * bn;           // b*nc + h within one action
    int b = (int)(r / nc), h = (int)(r % nc);
    out[t] = reach[r] * strat[(size_t)b * nact * nc + a * nc + h];
}

// Opponent node: dst[i] = sum over actions of utils[a][i].  (Bpn = B*pn)
__global__ void g_sum_utils(float* dst, const float* utils, int Bpn, int nact) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= Bpn) return;
    float s = 0.0f;
    for (int a = 0; a < nact; ++a) s += utils[(size_t)a * Bpn + t];
    dst[t] = s;
}

// Best-response player node: dst[i] = max over actions of utils[a][i]. One launch
// replacing g_fill(-inf) + nact g_max_b; algorithm-identical (the -1e30f seed makes
// the all-skipped case match the old fill). (Bpn = B*pn)
__global__ void g_max_all(float* dst, const float* utils, int Bpn, int nact) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= Bpn) return;
    float m = -1e30f;
    for (int a = 0; a < nact; ++a) m = fmaxf(m, utils[(size_t)a * Bpn + t]);
    dst[t] = m;
}

// Own-action node: dst[b*nc+h] = sum_a strat[(b*nact+a)*nc+h] * utils[a*B*nc + b*nc+h].
__global__ void g_fma_strat_all(float* dst, const float* strat, const float* utils, int nc, int nact, int B) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * nc) return;
    int b = t / nc, h = t % nc;
    const float* st = strat + (size_t)b * nact * nc;
    size_t bn = (size_t)B * nc;
    float s = 0.0f;
    for (int a = 0; a < nact; ++a) s += st[a * nc + h] * utils[(size_t)a * bn + t];
    dst[t] = s;
}

// Own-action node: DCFR update with the per-action regret computed inline from
// utils/pay (no materialized regret buffer). regret_a = utils[a*B*nc + t] - pay[t]
// (pn == nc here, so b*pn+h == t). Replaces nact g_set_regret_row_b + g_update_b.
__global__ void g_update_utils_b(const float* utils, const float* pay, __half* rplus, __half* cum,
                                 int nact, int nc, const float* coefs, float beta, float theta, int B) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * nc) return;
    int b = t / nc, h = t % nc;
    float alpha_coef = coefs[0], strat_coef = coefs[1];
    __half* rp = rplus + (size_t)b * nact * nc;
    __half* cm = cum + (size_t)b * nact * nc;
    size_t bn = (size_t)B * nc;
    float paysub = pay[t];
    float rsum = 0.0f;
    for (int a = 0; a < nact; ++a) {
        int idx = a * nc + h;
        float regret = utils[(size_t)a * bn + t] - paysub;
        float v = regret + __half2float(rp[idx]);
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

// DCFR discount coefficients for this iteration, kept device-resident so the
// captured CUDA graph stays static across iterations (only this 2-float buffer
// changes per replay). alpha=1.5, gamma=2 (beta/theta constants in g_update_utils_b).
__global__ void g_set_coefs(float* coefs, int iter) {
    float t = (float)(iter + 1);
    float a = powf(t, 1.5f);
    coefs[0] = a / (1.0f + a);                  // alpha_coef
    coefs[1] = powf(t / (t + 1.0f), 2.0f);      // strat_coef
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

// Product of per-level deal counts for chance levels 1..lvl = the batch B / trainset
// count at that depth. Mixed radix so flop iso (turn reduced, river full) works; for
// the uniform case nd_lvl1_==nd_lvl2_==ND, reproducing ND^lvl.
int CudaCfrSolver::Bprod(int lvl) const {
    if (lvl <= 0) return 1;
    if (lvl == 1) return nd_lvl1_;
    // lvl >= 2: with full-2 iso the river level is ragged, so the level-2 batch is
    // the total river-rep slot count (not a rectangular nd_lvl1_*nd_lvl2_ product).
    if (riv_iso_on_) return riv_total_;
    return nd_lvl1_ * nd_lvl2_;
}

// Trainable sets at an action node = one per compound runout reaching it.
int CudaCfrSolver::ntrainsets(int nodeid) const {
    return Bprod(level(nodeid));
}

// Checked device allocation: on failure report what failed, how much was needed,
// and how much VRAM remained, then throw. A full-range flop subgame's trainables
// are GB-scale; an unchecked cudaMalloc would otherwise return null and crash with
// a confusing device-side fault deep in a kernel instead of a clear OOM message.
static void* dmalloc(size_t bytes, const char* what) {
    void* p = nullptr;
    cudaError_t e = cudaMalloc(&p, bytes);
    if (e != cudaSuccess || p == nullptr) {
        size_t freeB = 0, totB = 0;
        cudaMemGetInfo(&freeB, &totB);
        char msg[256];
        snprintf(msg, sizeof(msg),
                 "cudaMalloc failed for %s: needed %.1f MB, only %.1f MB free of %.1f MB (%s)",
                 what, bytes / 1048576.0, freeB / 1048576.0, totB / 1048576.0,
                 cudaGetErrorString(e));
        throw std::runtime_error(msg);
    }
    return p;
}

CudaCfrSolver::CudaCfrSolver(const Subgame& sg, bool stream) : sg_(sg) {
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
            d_dealrank_[p] = (int*)dmalloc(sz * sizeof(int), "dealrank");
            cudaMemcpy(d_dealrank_[p], sg.dealrank[p].data(), sz * sizeof(int), cudaMemcpyHostToDevice);
        }
    }

    if (sg.has_chance && !sg.deal_cards.empty()) {
        size_t sz = sg.deal_cards.size();
        cudaMalloc(&d_deal_cards_, sz * sizeof(int));
        cudaMemcpy(d_deal_cards_, sg.deal_cards.data(), sz * sizeof(int), cudaMemcpyHostToDevice);
    }

    // Per-level deal sets. Default: every level is the full/deepest deal set (uniform
    // ND^level). The deepest level (river) is always d_deal_cards_; the shallower
    // turn level is overridden to the reduced representatives below for flop iso.
    if (sg.has_chance) {
        nd_lvl1_ = ND_;
        nd_lvl2_ = (sg.chance_levels >= 2) ? ND_ : 0;
        d_lvl1cards_ = d_deal_cards_;
        d_lvl2cards_ = (sg.chance_levels >= 2) ? d_deal_cards_ : nullptr;
    }

    // Suit isomorphism tables (turn): rep slot per full deal + per-player hand perm.
    if (sg.iso_on) {
        iso_on_ = true;
        iso_nd_full_ = sg.iso_nd_full;
        cudaMalloc(&d_iso_rep_slot_, (size_t)iso_nd_full_ * sizeof(int));
        cudaMemcpy(d_iso_rep_slot_, sg.iso_rep_slot.data(), (size_t)iso_nd_full_ * sizeof(int), cudaMemcpyHostToDevice);
        for (int p = 0; p < 2; p++) {
            size_t sz = sg.iso_perm[p].size();
            cudaMalloc(&d_iso_perm_[p], sz * sizeof(int));
            cudaMemcpy(d_iso_perm_[p], sg.iso_perm[p].data(), sz * sizeof(int), cudaMemcpyHostToDevice);
        }
        // Flop iso: the turn level deals only the representatives (river stays full in
        // d_lvl2cards_). Turn subgame leaves iso_level1_cards empty (its reduced level
        // is the deepest, already the reps in d_deal_cards_/d_lvl1cards_).
        if (!sg.iso_level1_cards.empty()) {
            nd_lvl1_ = (int)sg.iso_level1_cards.size();
            cudaMalloc(&d_lvl1cards_, (size_t)nd_lvl1_ * sizeof(int));
            cudaMemcpy(d_lvl1cards_, sg.iso_level1_cards.data(), (size_t)nd_lvl1_ * sizeof(int), cudaMemcpyHostToDevice);
        }
    }

    // Full 2-level iso (flop): ragged per-turn-rep river reps. The turn level keeps
    // its reduction (d_lvl1cards_ = turn reps above); here the river level batch is
    // riv_total_ slots and the level-2 chance uses the ragged expand/reduce kernels.
    if (sg.riv_iso_on) {
        riv_iso_on_ = true;
        riv_nt_ = sg.riv_nt;
        riv_total_ = sg.riv_total();
        d_riv_rep_cards_ = (int*)dmalloc((size_t)riv_total_ * sizeof(int), "riv rep cards");
        cudaMemcpy(d_riv_rep_cards_, sg.riv_rep_cards.data(), (size_t)riv_total_ * sizeof(int), cudaMemcpyHostToDevice);
        d_riv_slot2turn_ = (int*)dmalloc((size_t)riv_total_ * sizeof(int), "riv slot2turn");
        cudaMemcpy(d_riv_slot2turn_, sg.riv_slot2turn.data(), (size_t)riv_total_ * sizeof(int), cudaMemcpyHostToDevice);
        size_t fs = sg.riv_fullslot.size();
        d_riv_fullslot_ = (int*)dmalloc(fs * sizeof(int), "riv fullslot");
        cudaMemcpy(d_riv_fullslot_, sg.riv_fullslot.data(), fs * sizeof(int), cudaMemcpyHostToDevice);
        for (int p = 0; p < 2; p++) {
            size_t sz = sg.riv_perm[p].size();
            d_riv_perm_[p] = (int*)dmalloc(sz * sizeof(int), "riv perm");
            cudaMemcpy(d_riv_perm_[p], sg.riv_perm[p].data(), sz * sizeof(int), cudaMemcpyHostToDevice);
        }
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
            d_dealorder_[O] = (int*)dmalloc(order.size() * sizeof(int), "dealorder");
            cudaMemcpy(d_dealorder_[O], order.data(), order.size() * sizeof(int), cudaMemcpyHostToDevice);
            d_dealsortedranks_[O] = (int*)dmalloc(sranks.size() * sizeof(int), "dealsortedranks");
            cudaMemcpy(d_dealsortedranks_[O], sranks.data(), sranks.size() * sizeof(int), cudaMemcpyHostToDevice);
        }
    }

    int N = (int)sg.nodes.size();
    d_rplus_.assign(N, nullptr);
    d_cum_.assign(N, nullptr);
    nact_.assign(N, 0);
    h_rplus_.assign(N, nullptr);
    h_cum_.assign(N, nullptr);
    streamed_.assign(N, 0);

    // VRAM pre-flight. Trainables dominate the footprint (sets*nact*nc fp16, twice
    // for rplus+cum), and a full-range flop (sets=ND^2) is GB-scale. Estimate and
    // report up front; dmalloc below fails with a clear OOM message if it won't fit.
    size_t trainable_bytes = 0;
    for (int i = 0; i < N; ++i) {
        const Node& nd = sg.nodes[i];
        if (nd.type != NT_ACTION) continue;
        size_t sz = (size_t)ntrainsets(i) * nd.children.size() * ncards_[nd.player];
        trainable_bytes += sz * sizeof(__half) * 2;   // rplus + cum
    }
    size_t freeB = 0, totB = 0; cudaMemGetInfo(&freeB, &totB);

    // B-1 streaming decision. Only the river (level-2) nodes are huge and only the
    // plain non-iso 2-level flop has the streamable structure (the iso paths reduce
    // the batch by their own scheme). Auto-enable when the resident trainable would
    // not comfortably fit; `stream` forces it on (for validation on small spots).
    bool streamable = (sg.chance_levels == 2) && !iso_on_ && !riv_iso_on_;
    bool want_stream = streamable && (stream || trainable_bytes > (size_t)(freeB * 0.8));
    // The full trainable lives in host RAM while streaming. Refuse if it would not fit
    // available physical RAM with headroom — allocating GB-scale beyond RAM thrashes
    // and destabilizes the OS. Declining lets the caller's OOM->CPU fallback take over.
    size_t hostAvail = host_avail_bytes();
    size_t hostNeed = trainable_bytes + ((size_t)512 << 20);   // + ~0.5GB working slack
    bool hostOk = (hostAvail == 0) || (hostNeed + ((size_t)2 << 30) <= hostAvail);  // keep 2GB free
    stream_on_ = want_stream && hostOk;
    if (want_stream && !hostOk)
        printf("  NOTE: streaming needs ~%.1f GB host RAM, only ~%.1f GB available; NOT streaming"
               " (falls back to OOM->CPU).\n", hostNeed / 1073741824.0, hostAvail / 1073741824.0);

    printf("  VRAM: trainables ~%.0f MB (rplus+cum fp16) | %.0f MB free of %.0f MB%s\n",
           trainable_bytes / 1048576.0, freeB / 1048576.0, totB / 1048576.0,
           stream_on_ ? "  [STREAMING river trainables from host (pageable)]" : "");
    if (stream_on_ && hostAvail)
        printf("  host RAM: streaming %.1f GB of trainables (%.1f GB available)\n",
               trainable_bytes / 1073741824.0, hostAvail / 1073741824.0);

    for (int i = 0; i < N; ++i) {
        const Node& nd = sg.nodes[i];
        if (nd.type != NT_ACTION) continue;
        int nact = (int)nd.children.size();
        int nc = ncards_[nd.player];
        int sets = ntrainsets(i);
        nact_[i] = nact;
        // Stream the river (level-2) action nodes: full trainable in host-pinned RAM,
        // device buffer holds only one turn chunk (ND sets). Shallower (turn) nodes
        // are ND-times smaller, so keep them fully device-resident.
        if (stream_on_ && level(i) == 2) {
            streamed_[i] = 1;
            size_t full = (size_t)sets * nact * nc;          // ND^2 sets
            size_t chunk = (size_t)ND_ * nact * nc;          // one turn = ND sets
            // Pageable (swappable) host RAM, NOT pinned: the full trainable is GB-scale
            // and pinning that much page-locked memory starves and destabilizes the OS.
            // The synchronous chunk copies don't need pinning; future overlap would pin
            // only the small per-chunk staging buffers, never the whole trainable.
            h_rplus_[i] = new __half[full]();
            h_cum_[i] = new __half[full]();
            d_rplus_[i] = (__half*)dmalloc(chunk * sizeof(__half), "river chunk rplus"); cudaMemset(d_rplus_[i], 0, chunk * sizeof(__half));
            d_cum_[i] = (__half*)dmalloc(chunk * sizeof(__half), "river chunk cum"); cudaMemset(d_cum_[i], 0, chunk * sizeof(__half));
        } else {
            size_t sz = (size_t)sets * nact * nc;
            d_rplus_[i] = (__half*)dmalloc(sz * sizeof(__half), "trainable rplus"); cudaMemset(d_rplus_[i], 0, sz * sizeof(__half));
            d_cum_[i] = (__half*)dmalloc(sz * sizeof(__half), "trainable cum"); cudaMemset(d_cum_[i], 0, sz * sizeof(__half));
        }
    }

    // Per river-chance node, list its streamed descendants so each only loads/stores
    // its own chunk (one full-trainable pass per iteration, not one per chance node).
    stream_subtree_.assign(N, {});
    if (stream_on_)
        for (int i = 0; i < N; ++i)
            if (sg.nodes[i].type == NT_CHANCE && level(i) == 2)
                collectStreamed(i, stream_subtree_[i]);

    // LIFO scratch arena. Batched walk widens every scratch buffer by the node's
    // B = ND^level; the deepest level reaches ND^chance_levels. Peak usage is one
    // root-to-leaf path, tiny vs this floor for the modest test ranges. Streaming caps
    // the live batch at ND (one turn's rivers), so the arena floor shrinks accordingly.
    int maxnc = std::max(ncards_[0], ncards_[1]);
    int maxB = stream_on_ ? ND_ : Bprod(sg.chance_levels);
    size_t need = (size_t)maxnc * 64 * (size_t)maxB;
    arena_cap_ = std::max((size_t)256 * 1024 * 1024 / sizeof(float), need);
    arena_ = (float*)dmalloc(arena_cap_ * sizeof(float), "scratch arena");

    cudaStreamCreate(&stream_);
    cudaMalloc(&d_coefs_, 2 * sizeof(float));
}

CudaCfrSolver::~CudaCfrSolver() {
    for (int p = 0; p < 2; ++p) {
        cudaFree(d_c1_[p]); cudaFree(d_c2_[p]); cudaFree(d_rank_[p]);
        if (d_dealrank_[p]) cudaFree(d_dealrank_[p]);
    }
    if (d_deal_cards_) cudaFree(d_deal_cards_);
    if (d_lvl1cards_ && d_lvl1cards_ != d_deal_cards_) cudaFree(d_lvl1cards_);   // owned only for flop iso
    if (d_iso_rep_slot_) cudaFree(d_iso_rep_slot_);
    for (int p = 0; p < 2; p++) if (d_iso_perm_[p]) cudaFree(d_iso_perm_[p]);
    if (d_riv_rep_cards_) cudaFree(d_riv_rep_cards_);
    if (d_riv_slot2turn_) cudaFree(d_riv_slot2turn_);
    if (d_riv_fullslot_) cudaFree(d_riv_fullslot_);
    for (int p = 0; p < 2; p++) if (d_riv_perm_[p]) cudaFree(d_riv_perm_[p]);
    for (int O = 0; O < 2; ++O) {
        cudaFree(d_rankorder_[O]); cudaFree(d_sortedranks_[O]);
        cudaFree(d_cardoff_[O]); cudaFree(d_cardidx_[O]);
        if (d_dealorder_[O]) cudaFree(d_dealorder_[O]);
        if (d_dealsortedranks_[O]) cudaFree(d_dealsortedranks_[O]);
    }
    for (auto p : d_rplus_) cudaFree(p);
    for (auto p : d_cum_) cudaFree(p);
    for (auto p : h_rplus_) delete[] p;
    for (auto p : h_cum_) delete[] p;
    cudaFree(arena_);
    if (graph_exec_) cudaGraphExecDestroy(graph_exec_);
    if (d_coefs_) cudaFree(d_coefs_);
    cudaStreamDestroy(stream_);
}

// Bytes of one turn chunk (ND sets) of a streamed river node's regret/strategy.
size_t CudaCfrSolver::chunkSetBytes(int nodeid) const {
    return (size_t)ND_ * nact_[nodeid] * ncards_[sg_.nodes[nodeid].player] * sizeof(__half);
}

// Streamed action nodes reachable below `nodeid` (used to scope a river chance node's
// load/store to its own subtree). Chance nodes have a single child; recurse all.
void CudaCfrSolver::collectStreamed(int nodeid, std::vector<int>& out) const {
    const Node& nd = sg_.nodes[nodeid];
    if (nd.type == NT_ACTION && streamed_[nodeid]) out.push_back(nodeid);
    for (int c : nd.children) collectStreamed(c, out);
}

// Copy turn-deal t's ND-set slice of the given streamed river nodes from host into
// their device chunk buffers (regret + cumulative). Issued on stream_ so it orders
// with the subtree kernels that follow.
void CudaCfrSolver::streamLoadChunk(const std::vector<int>& nodes, int t) {
    for (int i : nodes) {
        size_t bytes = chunkSetBytes(i);
        size_t elemOff = (size_t)t * ND_ * nact_[i] * ncards_[sg_.nodes[i].player];
        cudaMemcpyAsync(d_rplus_[i], h_rplus_[i] + elemOff, bytes, cudaMemcpyHostToDevice, stream_);
        cudaMemcpyAsync(d_cum_[i], h_cum_[i] + elemOff, bytes, cudaMemcpyHostToDevice, stream_);
    }
}

// Copy the (now updated) chunk back to host after the turn's river subtree walk.
void CudaCfrSolver::streamStoreChunk(const std::vector<int>& nodes, int t) {
    for (int i : nodes) {
        size_t bytes = chunkSetBytes(i);
        size_t elemOff = (size_t)t * ND_ * nact_[i] * ncards_[sg_.nodes[i].player];
        cudaMemcpyAsync(h_rplus_[i] + elemOff, d_rplus_[i], bytes, cudaMemcpyDeviceToHost, stream_);
        cudaMemcpyAsync(h_cum_[i] + elemOff, d_cum_[i], bytes, cudaMemcpyDeviceToHost, stream_);
    }
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
    int B = Bprod(lvl);                       // batch = compound runouts reaching here
    // Streamed river walk: the subtree below the river chance runs one turn deal at a
    // time, so its batch is ND (one turn's rivers), not the full ND^2. The per-deal
    // rank/order tables are sliced to this turn's rows via deal_row_base_ (= turn*ND).
    if (streaming_active_ && lvl == 2) B = stream_b_;
    size_t drow = (size_t)deal_row_base_;     // dealrank/dealorder row offset (0 unless streamed)
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
            g_sd_scan_b<<<scan_grid, n2 / 2, n2 * sizeof(float), stream_>>>(d_reach, d_dealorder_[O] + drow * on, d_pref, on, n2);
            dim3 eval_grid(blocks_for((size_t)pn, T), B);
            g_sd_eval_b<<<eval_grid, T, 0, stream_>>>(d_dealrank_[player] + drow * pn, pn, d_pref, d_dealsortedranks_[O] + drow * on, on,
                d_c1_[player], d_c2_[player], d_cardoff_[O], d_cardidx_[O], d_dealrank_[1 - player] + drow * on, d_reach, win, lose, d_out);
            return;
        }
        const int* prk = use_dealrank ? d_dealrank_[player] + drow * pn : d_rank_[player];
        const int* ork = use_dealrank ? d_dealrank_[1 - player] + drow * on : d_rank_[1 - player];
        dim3 sd_grid(blocks_for((size_t)pn, T), B);
        size_t sd_shmem = (size_t)T * (3 * sizeof(int) + sizeof(float));
        g_showdown_b<<<sd_grid, T, sd_shmem, stream_>>>(d_c1_[player], d_c2_[player], prk, pn,
                                                        d_c1_[1 - player], d_c2_[1 - player], ork, d_reach, on,
                                                        win, lose, d_out, B);
        return;
    }
    if (nd.type == NT_TERMINAL) {
        float payoff = (float)nd.pay[player];
        // Streamed river terminal: b is just the river digit over the full river set,
        // so decode as level 1 (lvl1cards = river set) and exclude the turn card via
        // extra_card. Otherwise the normal compound-deal decode (extra_card = -1).
        const int* l1 = d_lvl1cards_; int n1 = nd_lvl1_;
        const int* l2 = d_lvl2cards_; int n2 = nd_lvl2_;
        int term_level = lvl, extra = -1, riv = riv_iso_on_ ? 1 : 0;
        if (streaming_active_ && lvl == 2) {
            l1 = d_deal_cards_; n1 = ND_; l2 = nullptr; n2 = 0;
            term_level = 1; extra = stream_turn_card_; riv = 0;
        }
        g_terminal_b<<<blocks_for((size_t)B * pn, T), T, 0, stream_>>>(d_c1_[player], d_c2_[player], pn,
                                                           d_c1_[1 - player], d_c2_[1 - player], d_reach, on,
                                                           payoff, l1, n1, l2, n2, term_level, d_out, B,
                                                           riv, d_riv_slot2turn_, d_riv_rep_cards_, extra);
        return;
    }
    if (nd.type == NT_CHANCE) {
        int child = nd.children[0];
        int out_level = lvl;                  // chance node's round = the round it deals into
        int in_level = out_level - 1;
        int Bin = Bprod(in_level);            // batch entering the chance
        float inv = 1.0f / (float)sg_.possible_deals[out_level - 1];

        // Full-2 iso: the level-2 (river) chance is ragged. Expand the NT turn-rep
        // reaches into riv_total_ river-rep slots, recurse once, then reduce each turn
        // rep's full rivers back via the per-rep hand permutation.
        if (riv_iso_on_ && out_level == 2) {
            size_t mark2 = arena_top_;
            float* d_nr2 = arena_alloc((size_t)riv_total_ * on);
            g_chance_expand_riv2<<<blocks_for((size_t)riv_total_ * on, T), T, 0, stream_>>>(
                d_reach, d_nr2, d_c1_[1 - player], d_c2_[1 - player],
                d_riv_slot2turn_, d_riv_rep_cards_, on, riv_total_, inv);
            float* d_cu2 = arena_alloc((size_t)riv_total_ * pn);
            cfr(player, child, d_nr2, d_cu2, br);     // B = riv_total_
            g_chance_reduce_riv2<<<blocks_for((size_t)Bin * pn, T), T, 0, stream_>>>(
                d_cu2, d_out, pn, Bin, ND_, d_riv_fullslot_, d_riv_perm_[player]);
            arena_top_ = mark2;
            return;
        }

        // Host-streamed river chance (non-iso flop): the entering batch Bin == ND is
        // one row per turn deal. Process turn deals one at a time so only ND river
        // trainsets are device-resident: load that turn's chunk, expand its reach into
        // ND river reaches, walk the river subtree with B=ND, reduce to this turn's
        // util row, store the chunk back. br mode keeps the full path (avg strategy is
        // resident there, only used on the small validation spots). Lossless.
        if (stream_on_ && out_level == 2 && !br) {
            const std::vector<int>& subtree = stream_subtree_[nodeid];
            for (int t = 0; t < Bin; ++t) {
                streamLoadChunk(subtree, t);
                int turncard = sg_.deal_cards[t];
                size_t markt = arena_top_;
                float* d_nr = arena_alloc((size_t)ND_ * on);
                g_chance_expand_stream<<<blocks_for((size_t)ND_ * on, T), T, 0, stream_>>>(
                    d_reach + (size_t)t * on, d_nr, d_c1_[1 - player], d_c2_[1 - player],
                    d_deal_cards_, ND_, turncard, on, inv);
                float* d_cu = arena_alloc((size_t)ND_ * pn);
                streaming_active_ = true; stream_b_ = ND_; deal_row_base_ = t * ND_; stream_turn_card_ = turncard;
                cfr(player, child, d_nr, d_cu, br);
                streaming_active_ = false; deal_row_base_ = 0; stream_turn_card_ = -1;
                g_chance_reduce<<<blocks_for((size_t)pn, T), T, 0, stream_>>>(d_cu, d_out + (size_t)t * pn, pn, ND_, 1);
                streamStoreChunk(subtree, t);
                arena_top_ = markt;
            }
            return;
        }

        int nd_cur = (out_level >= 2) ? nd_lvl2_ : nd_lvl1_;          // deals at this level
        const int* curcards = (out_level >= 2) ? d_lvl2cards_ : d_lvl1cards_;
        size_t mark = arena_top_;
        float* d_nr = arena_alloc((size_t)Bin * nd_cur * on);
        g_chance_expand<<<blocks_for((size_t)Bin * nd_cur * on, T), T, 0, stream_>>>(d_reach, d_nr,
                        d_c1_[1 - player], d_c2_[1 - player], curcards, nd_cur, d_lvl1cards_, nd_lvl1_, in_level, on, Bin, inv);
        float* d_cu = arena_alloc((size_t)Bin * nd_cur * pn);
        cfr(player, child, d_nr, d_cu, br);       // child one level deeper => B=Bin*nd_cur
        // Level-1 iso: re-expand the reduced reps to all real runouts via the hand
        // permutation. Deeper levels (river under a flop) stay full -> plain reduce.
        if (iso_on_ && out_level == 1)
            g_chance_reduce_iso<<<blocks_for((size_t)Bin * pn, T), T, 0, stream_>>>(
                d_cu, d_out, pn, nd_cur, Bin, iso_nd_full_, d_iso_rep_slot_, d_iso_perm_[player]);
        else
            g_chance_reduce<<<blocks_for((size_t)Bin * pn, T), T, 0, stream_>>>(d_cu, d_out, pn, nd_cur, Bin);
        arena_top_ = mark;
        return;
    }

    // ACTION node
    int np = nd.player;
    int nact = nact_[nodeid];
    int nc = ncards_[np];

    if (br) {
        // Best response: cfr-player maxes over actions; opponent plays avg strategy.
        // Same fused per-action kernels as the train path (g_row_mul_all/g_sum_utils),
        // with g_max_all for the BR player's max — algorithm-identical to the old
        // per-action g_fill/g_max_b/g_row_mul_b/g_add loops, fewer launches.
        size_t mark = arena_top_;
        float* utils = arena_alloc((size_t)nact * B * pn);
        if (np == player) {                         // BR player: reach unchanged, max
            for (int a = 0; a < nact; ++a)
                cfr(player, nd.children[a], d_reach, utils + (size_t)a * B * pn, true);
            g_max_all<<<blocks_for((size_t)B * pn, T), T, 0, stream_>>>(d_out, utils, (int)((size_t)B * pn), nact);
        } else {                                    // opponent: avg-strategy reach, sum
            float* avg = d_avgstrat_[nodeid];       // [B*nact*nc] average strategy
            float* d_newreach = arena_alloc((size_t)nact * B * nc);   // nc == on here
            g_row_mul_all<<<blocks_for((size_t)nact * B * nc, T), T, 0, stream_>>>(d_reach, avg, d_newreach, nc, nact, B);
            for (int a = 0; a < nact; ++a)
                cfr(player, nd.children[a], d_newreach + (size_t)a * B * nc, utils + (size_t)a * B * pn, true);
            g_sum_utils<<<blocks_for((size_t)B * pn, T), T, 0, stream_>>>(d_out, utils, (int)((size_t)B * pn), nact);
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
    if (np != player) {
        // Opponent node: scale reach by every action's strategy in one launch,
        // recurse each child, then sum the action utilities.
        float* d_newreach = arena_alloc((size_t)nact * B * nc);   // nc == on here
        g_row_mul_all<<<blocks_for((size_t)nact * B * nc, T), T, 0, stream_>>>(d_reach, d_strat, d_newreach, nc, nact, B);
        for (int a = 0; a < nact; ++a)
            cfr(player, nd.children[a], d_newreach + (size_t)a * B * nc, utils + (size_t)a * B * pn);
        g_sum_utils<<<blocks_for((size_t)B * pn, T), T, 0, stream_>>>(d_out, utils, (int)((size_t)B * pn), nact);
    } else {
        // Own-action node: reach unchanged; strategy-weight the action utilities
        // and run the fused DCFR update (regret computed inline from utils/pay).
        for (int a = 0; a < nact; ++a)
            cfr(player, nd.children[a], d_reach, utils + (size_t)a * B * pn);
        g_fma_strat_all<<<blocks_for((size_t)B * nc, T), T, 0, stream_>>>(d_out, d_strat, utils, nc, nact, B);
        g_update_utils_b<<<blocks_for((size_t)B * nc, T), T, 0, stream_>>>(utils, d_out, rplus, cum, nact, nc,
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

    // Streaming mode replays a host-side loop over turn chunks with interleaved
    // host<->device copies whose offsets change every chunk, which a single static
    // CUDA graph cannot capture. The streamed flop is compute-bound, so running the
    // walk eagerly (no graph) costs nothing measurable. Other modes capture once.
    std::chrono::high_resolution_clock::time_point t0, t1;
    if (stream_on_) {
        cudaDeviceSynchronize();
        t0 = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < iters; ++it) {
            g_set_coefs<<<1, 1, 0, stream_>>>(d_coefs_, it);
            runIteration();
        }
        cudaStreamSynchronize(stream_);
        t1 = std::chrono::high_resolution_clock::now();
    } else {
        // Capture one iteration's kernel stream into a replayable graph.
        cudaGraph_t graph = nullptr;
        cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal);
        runIteration();
        cudaStreamEndCapture(stream_, &graph);
        cudaGraphInstantiate(&graph_exec_, graph, nullptr, nullptr, 0);
        cudaGraphDestroy(graph);

        cudaDeviceSynchronize();
        t0 = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < iters; ++it) {
            g_set_coefs<<<1, 1, 0, stream_>>>(d_coefs_, it);   // per-iter, outside the graph
            cudaGraphLaunch(graph_exec_, stream_);
        }
        cudaStreamSynchronize(stream_);
        t1 = std::chrono::high_resolution_clock::now();
    }

    cudaFree(d_init_[0]); cudaFree(d_init_[1]);
    d_init_[0] = d_init_[1] = nullptr;

    // Regrets are no longer needed (averaging/exploitability read only d_cum_).
    // Release them so the post-training peak (cum fp16 + fp32 avg strategy) is
    // smaller — the binding constraint for large flop subgames.
    freeRegrets();
    return std::chrono::duration<double>(t1 - t0).count();
}

void CudaCfrSolver::freeRegrets() {
    for (auto& p : d_rplus_) { if (p) { cudaFree(p); p = nullptr; } }
    for (auto& p : h_rplus_) { if (p) { delete[] p; p = nullptr; } }   // host full regret (streamed)
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
        size_t setSz = (size_t)nact * nc;
        size_t sz = (size_t)sets * setSz;
        out[i].resize(sz);
        if (streamed_[i]) {
            // cum lives in host RAM ([ND^2 sets]); normalize one turn chunk (ND sets)
            // at a time through the resident device chunk buffer, mirroring training.
            size_t chunkElems = (size_t)ND_ * setSz;
            float* d_avg = (float*)dmalloc(chunkElems * sizeof(float), "avg-strategy chunk");
            for (int t = 0; t < ND_; ++t) {
                cudaMemcpy(d_cum_[i], h_cum_[i] + (size_t)t * chunkElems, chunkElems * sizeof(__half), cudaMemcpyHostToDevice);
                for (int s = 0; s < ND_; ++s)
                    g_avg<<<blocks_for(nc, T), T>>>(d_cum_[i] + (size_t)s * setSz, d_avg + (size_t)s * setSz, nact, nc);
                cudaDeviceSynchronize();
                cudaMemcpy(out[i].data() + (size_t)t * chunkElems, d_avg, chunkElems * sizeof(float), cudaMemcpyDeviceToHost);
            }
            cudaFree(d_avg);
            continue;
        }
        float* d_avg = (float*)dmalloc(sz * sizeof(float), "avg-strategy scratch");
        for (int s = 0; s < sets; ++s)
            g_avg<<<blocks_for(nc, T), T>>>(d_cum_[i] + (size_t)s * setSz,
                                            d_avg + (size_t)s * setSz, nact, nc);
        cudaDeviceSynchronize();
        cudaMemcpy(out[i].data(), d_avg, sz * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_avg);
    }
    return out;
}

// One action node's average strategy for a single trainset slot ([nact*nc], layout
// a*nc+h), normalized per hand like g_avg (avg = cum / sum_a cum, uniform if zero).
// Reads only that set's cumulative — host for streamed nodes, device otherwise — so a
// single spot costs nact*nc, not the node's full ND^level trainable.
std::vector<float> CudaCfrSolver::averageStrategyForSet(int nodeid, int slot) const {
    int nact = nact_[nodeid];
    int nc = ncards_[sg_.nodes[nodeid].player];
    size_t setElems = (size_t)nact * nc;
    std::vector<__half> cum(setElems);
    if (streamed_[nodeid]) {
        memcpy(cum.data(), h_cum_[nodeid] + (size_t)slot * setElems, setElems * sizeof(__half));
    } else {
        cudaMemcpy(cum.data(), d_cum_[nodeid] + (size_t)slot * setElems, setElems * sizeof(__half), cudaMemcpyDeviceToHost);
    }
    std::vector<float> avg(setElems);
    for (int h = 0; h < nc; h++) {
        float c = 0.0f;
        for (int a = 0; a < nact; a++) c += __half2float(cum[(size_t)a * nc + h]);
        for (int a = 0; a < nact; a++) {
            size_t idx = (size_t)a * nc + h;
            avg[idx] = (c > 0.0f) ? __half2float(cum[idx]) / c : (1.0f / nact);
        }
    }
    return avg;
}

// Every action node's average strategy at a single runout. runout holds the dealt-card
// indices most-significant first (turn then river); a node at chance depth `level` uses
// the first `level` of them as a base-ND compound slot. Non-iso layout only.
std::vector<std::vector<float>> CudaCfrSolver::averageStrategiesForRunout(const std::vector<int>& runout) const {
    int N = (int)sg_.nodes.size();
    std::vector<std::vector<float>> out(N);
    for (int i = 0; i < N; i++) {
        if (sg_.nodes[i].type != NT_ACTION) continue;
        int lvl = level(i);
        int slot = 0;
        for (int k = 0; k < lvl; k++) slot = slot * ND_ + runout[k];   // base-ND, turn before river
        out[i] = averageStrategyForSet(i, slot);
    }
    return out;
}

double CudaCfrSolver::exploitability() {
    int N = (int)sg_.nodes.size();
    // Upload the average strategy per action node for the best-response traversal.
    std::vector<std::vector<float>> avgs = averageStrategies();
    d_avgstrat_.assign(N, nullptr);
    // The BR path is not streamed, so it needs the full ND^2 avg strategy resident as
    // fp32 (2x the fp16 trainable). For a large streamed flop that does not fit; the
    // metric is validation-only, so skip it cleanly rather than abort the solve.
    try {
        for (int i = 0; i < N; ++i) {
            if (sg_.nodes[i].type != NT_ACTION) continue;
            size_t sz = avgs[i].size();
            d_avgstrat_[i] = (float*)dmalloc(sz * sizeof(float), "best-response avg strategy");
            cudaMemcpy(d_avgstrat_[i], avgs[i].data(), sz * sizeof(float), cudaMemcpyHostToDevice);
        }
    } catch (const std::exception& e) {
        for (int i = 0; i < N; ++i) if (d_avgstrat_[i]) { cudaFree(d_avgstrat_[i]); d_avgstrat_[i] = nullptr; }
        d_avgstrat_.clear();
        printf("exploitability: skipped (avg strategy exceeds VRAM under streaming: %s)\n", e.what());
        return -1.0;
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
