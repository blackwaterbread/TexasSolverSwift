#pragma once
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "subgame.h"

namespace texgpu {

// GPU Discounted-CFR solver for a fixed river subgame (host-driven tree walk,
// per-node kernels). Mirrors PCfrSolver's algorithm.
class CudaCfrSolver {
public:
    explicit CudaCfrSolver(const Subgame& sg);
    ~CudaCfrSolver();

    // Runs `iters` DCFR iterations. Returns wall-clock seconds spent in the loop.
    double train(int iters);

    // Average strategy per action node, indexed by node id; each entry has size
    // ntrainsets(node)*nact*ncards(node.player), layout (slot*nact + action)*nc + hand.
    // ntrainsets = ND^level (one set per compound runout). Non-action nodes empty.
    std::vector<std::vector<float>> averageStrategies();

    // Exploitability of the current average strategy (equilibrium-invariant; ->0
    // as the solve converges to Nash). Returns chips/matchup; also prints %pot.
    double exploitability();

    int ntrainsets(int nodeid) const;

private:
    const Subgame& sg_;

    // device range arrays per player
    int* d_c1_[2] = {nullptr, nullptr};
    int* d_c2_[2] = {nullptr, nullptr};
    int* d_rank_[2] = {nullptr, nullptr};       // base ranks (river root)
    int* d_dealrank_[2] = {nullptr, nullptr};   // per-deal ranks (ndeals*ncards), turn
    int* d_deal_cards_ = nullptr;               // runout card per deal (ndeals), turn
    // O(n) showdown (card-sum trick) static structures, per range side (river only):
    int* d_rankorder_[2] = {nullptr, nullptr};  // combo indices sorted by base rank
    int* d_sortedranks_[2] = {nullptr, nullptr};// those ranks (for binary search)
    int* d_cardoff_[2] = {nullptr, nullptr};    // CSR offsets [53]: hands containing each card
    int* d_cardidx_[2] = {nullptr, nullptr};    // CSR values [2*ncombos]: combo indices
    // per-deal rank-sorted order/ranks for the batched O(n) showdown (chance subgames)
    int* d_dealorder_[2] = {nullptr, nullptr};      // [nrows*ncombos] combo idx sorted per deal
    int* d_dealsortedranks_[2] = {nullptr, nullptr};// [nrows*ncombos] those ranks
    int ncards_[2] = {0, 0};
    int root_round_ = 3;
    int ND_ = 1;                                // deal-card count (max(1, ndeals))

    // Suit isomorphism (turn subgame). When on, ND_ holds the representative count;
    // the chance node deals representatives and g_chance_reduce_iso re-expands to all
    // iso_nd_full_ real runouts via the per-player hand permutation. Off => plain reduce.
    bool iso_on_ = false;
    int iso_nd_full_ = 0;
    int* d_iso_rep_slot_ = nullptr;             // [nd_full] representative slot per full deal
    int* d_iso_perm_[2] = {nullptr, nullptr};   // [nd_full * ncards_[player]] hand permutation

    // per action-node persistent device state (fp16 storage / fp32 compute, like
    // the CPU HF trainable; halves the dominant trainable memory). null if non-action.
    std::vector<__half*> d_rplus_;
    std::vector<__half*> d_cum_;
    std::vector<int> nact_;   // per node

    int level(int nodeid) const;   // chance depth = node.round - root_round

    // Frees the per-node regret buffers (d_rplus_). They are only needed during
    // training; averageStrategies()/exploitability() read d_cum_ (and a transient
    // fp32 avg), so releasing them after train() lowers the post-training VRAM
    // peak — the lever that matters for GB-scale flop subgames. Idempotent.
    void freeRegrets();

    // Writes payoff into caller-provided d_out. Buffers carry a batch dimension
    // B = ND^level (1 above all chances): d_reach is [B*ncards(opp)], d_out is
    // [B*ncards(player)]. The batch index equals the compound deal/trainset slot.
    // All work issues on stream_ for CUDA-graph capture.
    void cfr(int player, int nodeid, const float* d_reach, float* d_out, bool br = false);
    void runIteration();           // one DCFR pass (both players); the captured body

    // device average strategies per action node (float [sets*nact*nc]); populated
    // only during exploitability() for the best-response traversal.
    std::vector<float*> d_avgstrat_;

    // LIFO device scratch arena (bump allocator).
    float* arena_ = nullptr;
    size_t arena_cap_ = 0;   // in floats
    size_t arena_top_ = 0;   // in floats
    float* arena_alloc(size_t n);

    // CUDA-graph replay: the per-iteration kernel sequence is identical, so it is
    // captured once and replayed. d_init_ holds root reach probs; d_coefs_ holds
    // the per-iteration DCFR discount pair (set before each replay).
    cudaStream_t stream_ = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;
    float* d_init_[2] = {nullptr, nullptr};
    float* d_coefs_ = nullptr;     // [alpha_coef, strat_coef]
};

} // namespace texgpu
