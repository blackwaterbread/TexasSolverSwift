#pragma once

// Smoke-test kernel launcher. Real CFR kernels (regret-matching, DCFR update,
// showdown/terminal utility) will be added here in later phases.
namespace texgpu {

// out = a*x + y, element-wise over n elements. Returns true on success.
bool saxpy_check(float a, int n);

// Prints the active GPU name and compute capability. Returns false if no device.
bool print_device_info();

// --- Leaf utility kernels (Phase 3) ---
// All arrays are Structure-of-Arrays. Player payoff vector is indexed by the
// player's range; opponent reach is indexed by the opponent's range.
// rank: smaller = stronger. Hands sharing any card contribute nothing (card removal).

// Showdown: out[i] = sum_j disjoint(i,j) * reach[j] *
//   (rank_i<rank_j ? win_payoff : rank_i>rank_j ? lose_payoff : 0)
void showdown_payoff_gpu(
    const int* p_c1, const int* p_c2, const int* p_rank, int pn,
    const int* o_c1, const int* o_c2, const int* o_rank, const float* o_reach, int on,
    float win_payoff, float lose_payoff, float* out_host);

// Terminal (fold): out[i] = payoff * sum_j disjoint(i,j) * reach[j]
void terminal_payoff_gpu(
    const int* p_c1, const int* p_c2, int pn,
    const int* o_c1, const int* o_c2, const float* o_reach, int on,
    float payoff, float* out_host);

// --- Discounted-CFR trainable (Phase 4) ---
// Runs `iters` DCFR updates for a single infoset on the GPU. regret_seq has
// `iters` rows, each of length nact*ncards (indexed action*ncards + hand).
// Writes the final average strategy (same layout) to out_avg_host.
// Mirrors DiscountedCfrTrainable (alpha=1.5, beta=0.5, gamma=2, theta=0.9).
void dcfr_run_gpu(int nact, int ncards, int iters,
                  const float* regret_seq, float* out_avg_host);

} // namespace texgpu
