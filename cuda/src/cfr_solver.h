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
    // stream: force host-streaming of the river (level-2) trainables (large non-iso
    // flop). When false the ctor still auto-enables it if the estimated trainable
    // footprint would not fit in VRAM (and the subgame is a streamable non-iso flop).
    explicit CudaCfrSolver(const Subgame& sg, bool stream = false);
    ~CudaCfrSolver();

    bool streaming() const { return stream_on_; }

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

    // Full 2-level iso (flop): the river level is reduced per turn rep (RAGGED). The
    // level-2 batch is riv_total_ absolute river-rep slots (sum over turn reps of
    // their river-rep counts); turn rep t owns slots given by the host-side offsets.
    // The level-2 chance expands the NT turn-rep reaches into these slots and reduces
    // them back by summing each turn rep's full rivers via the per-rep hand perm.
    bool riv_iso_on_ = false;
    int riv_nt_ = 0;                             // turn rep count (NT) = Bin at level 2
    int riv_total_ = 0;                          // total river-rep slots (level-2 batch)
    int* d_riv_rep_cards_ = nullptr;            // [riv_total_] river rep card per abs slot
    int* d_riv_slot2turn_ = nullptr;            // [riv_total_] turn rep index per abs slot
    int* d_riv_fullslot_ = nullptr;             // [NT * ND] (turn rep, full river) -> abs slot
    int* d_riv_perm_[2] = {nullptr, nullptr};   // [NT * ND * ncards_[player]] hand relabel

    // Per-chance-level deal sets (mixed radix; at most 2 levels: flop deals turn then
    // river). nd_lvl1_/nd_lvl2_ are the deal counts and d_lvl1cards_/d_lvl2cards_ the
    // card ints at each level. Uniform (no iso): every level is the full deal set, so
    // Bprod reproduces the old ND^level batching. With turn iso the (single) level is
    // reduced; with flop level-1 iso the turn level is reduced while river stays full.
    int nd_lvl1_ = 1, nd_lvl2_ = 0;
    int* d_lvl1cards_ = nullptr;
    int* d_lvl2cards_ = nullptr;
    int Bprod(int lvl) const;                   // product of per-level deal counts for levels 1..lvl

    // per action-node persistent device state (fp16 storage / fp32 compute, like
    // the CPU HF trainable; halves the dominant trainable memory). null if non-action.
    // For host-streamed river nodes (streamed_[i]) these hold only the CURRENT turn
    // chunk ([ND*nact*nc]); the full [ND^2*nact*nc] lives in h_rplus_/h_cum_.
    std::vector<__half*> d_rplus_;
    std::vector<__half*> d_cum_;
    std::vector<int> nact_;   // per node

    // --- B-1 host streaming of river (level-2) trainables (large non-iso flop) ---
    // The persistent river trainable is the VRAM wall (ND^2 sets). When streaming,
    // each river action node keeps its full regret/strategy in host-pinned memory and
    // the solve runs one turn deal at a time: load that turn's ND-set chunk into the
    // device chunk buffer (d_rplus_/d_cum_), walk the river subtree with B=ND, store
    // the chunk back. The flop is compute-bound, so the copies hide under compute.
    // Lossless (per-set DCFR math unchanged). Non-iso 2-level flop only.
    bool stream_on_ = false;              // streaming enabled (auto or forced)
    bool streaming_active_ = false;       // currently inside the per-chunk river walk
    int stream_b_ = 0;                    // B override for the streamed subtree (== ND)
    int deal_row_base_ = 0;               // dealrank/dealorder row offset (turn t * ND)
    int stream_turn_card_ = -1;           // turn-deal card excluded by streamed terminals
    std::vector<__half*> h_rplus_;        // [N] host-pinned full river regret (streamed)
    std::vector<__half*> h_cum_;          // [N] host-pinned full river cumulative (streamed)
    std::vector<char> streamed_;          // [N] 1 if this node's trainable is host-streamed
    // Per river-chance node (out_level==2): the streamed action nodes in its subtree.
    // Each streamed node belongs to exactly one river chance, so loading/storing only
    // a chance node's own subtree keeps the per-iteration host<->device traffic at one
    // pass over the full trainable (not one pass per river-chance node).
    std::vector<std::vector<int>> stream_subtree_;
    void collectStreamed(int nodeid, std::vector<int>& out) const;
    size_t chunkSetBytes(int nodeid) const;   // bytes of one turn chunk's ND sets
    void streamLoadChunk(const std::vector<int>& nodes, int t);   // turn t chunk host->device
    void streamStoreChunk(const std::vector<int>& nodes, int t);  // turn t chunk device->host

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
