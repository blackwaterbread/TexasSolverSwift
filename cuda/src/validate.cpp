#include "validate.h"
#include "cuda_kernels.cuh"

#include <cstdio>
#include <cmath>
#include <vector>

namespace texgpu {

static inline bool disjoint_h(int a1, int a2, int b1, int b2) {
    return a1 != b1 && a1 != b2 && a2 != b1 && a2 != b2;
}

// CPU brute-force references (mirror the kernels exactly).
static std::vector<float> showdown_cpu(
    const std::vector<Combo>& P, const std::vector<Combo>& O,
    const std::vector<float>& reach, float win, float lose) {
    std::vector<float> out(P.size(), 0.0f);
    for (size_t i = 0; i < P.size(); ++i) {
        float acc = 0.0f;
        for (size_t j = 0; j < O.size(); ++j) {
            if (!disjoint_h(P[i].card1, P[i].card2, O[j].card1, O[j].card2)) continue;
            if (P[i].rank < O[j].rank) acc += win * reach[j];
            else if (P[i].rank > O[j].rank) acc += lose * reach[j];
        }
        out[i] = acc;
    }
    return out;
}

static std::vector<float> terminal_cpu(
    const std::vector<Combo>& P, const std::vector<Combo>& O,
    const std::vector<float>& reach, float payoff) {
    std::vector<float> out(P.size(), 0.0f);
    for (size_t i = 0; i < P.size(); ++i) {
        float acc = 0.0f;
        for (size_t j = 0; j < O.size(); ++j)
            if (disjoint_h(P[i].card1, P[i].card2, O[j].card1, O[j].card2)) acc += reach[j];
        out[i] = payoff * acc;
    }
    return out;
}

// Deterministic pseudo-random reach in [0,1) (LCG), so runs are reproducible.
static std::vector<float> make_reach(int n, uint32_t seed) {
    std::vector<float> r(n);
    uint32_t s = seed ? seed : 1u;
    for (int i = 0; i < n; ++i) {
        s = 1664525u * s + 1013904223u;
        r[i] = (float)((s >> 8) & 0xFFFFFF) / (float)0x1000000;
    }
    return r;
}

static void soa(const std::vector<Combo>& c, std::vector<int>& c1, std::vector<int>& c2,
                std::vector<int>& rk) {
    c1.resize(c.size()); c2.resize(c.size()); rk.resize(c.size());
    for (size_t i = 0; i < c.size(); ++i) { c1[i] = c[i].card1; c2[i] = c[i].card2; rk[i] = c[i].rank; }
}

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

bool validate_leaves(const Subgame& sg, float tol) {
    int checks = 0, failures = 0;
    float worst = 0.0f;

    std::vector<int> pc1, pc2, prk, oc1, oc2, ork;

    for (size_t nid = 0; nid < sg.nodes.size(); ++nid) {
        const Node& nd = sg.nodes[nid];
        if (nd.type == NT_ACTION) continue;

        for (int player = 0; player < 2; ++player) {
            int oppo = 1 - player;
            const auto& P = sg.ranges[player];
            const auto& O = sg.ranges[oppo];
            soa(P, pc1, pc2, prk);
            soa(O, oc1, oc2, ork);
            std::vector<float> reach = make_reach((int)O.size(), (uint32_t)(nid * 7 + player + 1));

            std::vector<float> cpu, gpu(P.size(), 0.0f);
            if (nd.type == NT_SHOWDOWN) {
                float win = (float)nd.sd[player][player];
                float lose = (float)nd.sd[oppo][player];
                cpu = showdown_cpu(P, O, reach, win, lose);
                showdown_payoff_gpu(pc1.data(), pc2.data(), prk.data(), (int)P.size(),
                                    oc1.data(), oc2.data(), ork.data(), reach.data(), (int)O.size(),
                                    win, lose, gpu.data());
            } else { // NT_TERMINAL
                float payoff = (float)nd.pay[player];
                cpu = terminal_cpu(P, O, reach, payoff);
                terminal_payoff_gpu(pc1.data(), pc2.data(), (int)P.size(),
                                    oc1.data(), oc2.data(), reach.data(), (int)O.size(),
                                    payoff, gpu.data());
            }

            float d = max_abs_diff(cpu, gpu);
            worst = std::max(worst, d);
            checks++;
            if (d > tol) {
                failures++;
                printf("  MISMATCH node=%zu type=%d player=%d max|diff|=%g\n",
                       nid, (int)nd.type, player, d);
            }
        }
    }
    printf("leaf validation: %d checks, %d failures, worst |diff|=%g (tol=%g)\n",
           checks, failures, worst, tol);
    return failures == 0;
}

// CPU mirror of DiscountedCfrTrainable, returns final average strategy.
static std::vector<float> dcfr_run_cpu(int nact, int ncards, int iters,
                                       const std::vector<float>& regret_seq) {
    const float beta = 0.5f, gamma = 2.0f, theta = 0.9f, alpha = 1.5f;
    int sz = nact * ncards;
    std::vector<float> r_plus(sz, 0.0f), cum(sz, 0.0f);

    for (int it = 0; it < iters; ++it) {
        int t = it + 1;
        float alpha_coef = std::pow((float)t, alpha);
        alpha_coef = alpha_coef / (1.0f + alpha_coef);
        float strat_coef = std::pow((float)t / (float)(t + 1), gamma);
        const float* reg = regret_seq.data() + (size_t)it * sz;

        for (int h = 0; h < ncards; ++h) {
            float rsum = 0.0f;
            for (int a = 0; a < nact; ++a) {
                int idx = a * ncards + h;
                float v = reg[idx] + r_plus[idx];
                v = (v > 0.0f) ? v * alpha_coef : v * beta;
                r_plus[idx] = v;
                if (v > 0.0f) rsum += v;
            }
            for (int a = 0; a < nact; ++a) {
                int idx = a * ncards + h;
                float s = (rsum > 0.0f) ? (r_plus[idx] > 0.0f ? r_plus[idx] / rsum : 0.0f)
                                        : (1.0f / nact);
                cum[idx] = cum[idx] * theta + s * strat_coef;
            }
        }
    }
    std::vector<float> avg(sz, 0.0f);
    for (int h = 0; h < ncards; ++h) {
        float csum = 0.0f;
        for (int a = 0; a < nact; ++a) csum += cum[a * ncards + h];
        for (int a = 0; a < nact; ++a) {
            int idx = a * ncards + h;
            avg[idx] = (csum > 0.0f) ? cum[idx] / csum : (1.0f / nact);
        }
    }
    return avg;
}

bool validate_trainable(int nact, int ncards, int iters, float tol) {
    int sz = nact * ncards;
    std::vector<float> regret_seq((size_t)iters * sz);
    // deterministic pseudo-random regrets in [-1,1)
    uint32_t s = 12345u;
    for (size_t i = 0; i < regret_seq.size(); ++i) {
        s = 1664525u * s + 1013904223u;
        regret_seq[i] = ((float)((s >> 8) & 0xFFFFFF) / (float)0x1000000) * 2.0f - 1.0f;
    }

    std::vector<float> cpu = dcfr_run_cpu(nact, ncards, iters, regret_seq);
    std::vector<float> gpu(sz, 0.0f);
    dcfr_run_gpu(nact, ncards, iters, regret_seq.data(), gpu.data());

    float worst = max_abs_diff(cpu, gpu);
    printf("trainable validation: nact=%d ncards=%d iters=%d worst |diff|=%g (tol=%g)\n",
           nact, ncards, iters, worst, tol);
    return worst <= tol;
}

} // namespace texgpu
