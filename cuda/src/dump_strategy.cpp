#include "dump_strategy.h"

#include <fstream>
#include <stdexcept>
#include "json.hpp"

using json = nlohmann::json;

namespace texgpu {

// Builds the per-hand strategy map for one trainset slot of an action node.
// `slot` indexes the compound runout; layout is (slot*nact + action)*nc + hand.
static json strat_map_for_slot(const Subgame& sg, const Node& nd,
                               const std::vector<float>& av,
                               int slot, int nact, int nc) {
    json jstrat = json::object();
    size_t base = (size_t)slot * nact * nc;
    for (int h = 0; h < nc; h++) {
        json probs = json::array();
        for (int a = 0; a < nact; a++) {
            size_t idx = base + (size_t)a * nc + h;
            float v = (idx < av.size()) ? av[idx] : 0.0f;
            probs.push_back(v);
        }
        jstrat[sg.ranges[nd.player][h].label] = probs;
    }
    return jstrat;
}

// Iso variant: the trainable only stores representative runouts, so a full runout's
// strategy for hand h is the representative's strategy for the suit-swapped hand
// perm[h] (slot = rep_slot). Recovers the full per-runout strategy from the rep.
static json strat_map_for_full_deal(const Subgame& sg, const Node& nd,
                                    const std::vector<float>& av,
                                    int full_deal, int nact, int nc) {
    const int* perm = sg.iso_perm[nd.player].data() + (size_t)full_deal * nc;
    size_t base = (size_t)sg.iso_rep_slot[full_deal] * nact * nc;
    json jstrat = json::object();
    for (int h = 0; h < nc; h++) {
        int ph = perm[h];
        json probs = json::array();
        for (int a = 0; a < nact; a++) {
            size_t idx = base + (size_t)a * nc + ph;
            float v = (idx < av.size()) ? av[idx] : 0.0f;
            probs.push_back(v);
        }
        jstrat[sg.ranges[nd.player][h].label] = probs;
    }
    return jstrat;
}

// Flop iso variant (2 levels): the trainable slot is turn_rep*ND + river, with the
// river remapped by the full turn card's suit swap (iso_riverperm) and the hands
// relabeled by the same swap (iso_perm). Recovers a full (turn, river) runout's
// strategy from the representative-turn trainable.
static json strat_map_for_flop_deal(const Subgame& sg, const Node& nd,
                                    const std::vector<float>& av,
                                    int turn_full, int river, int ND, int nact, int nc) {
    const int* perm = sg.iso_perm[nd.player].data() + (size_t)turn_full * nc;
    int river_rep = sg.iso_riverperm[(size_t)turn_full * ND + river];
    size_t base = ((size_t)sg.iso_rep_slot[turn_full] * ND + river_rep) * nact * nc;
    json jstrat = json::object();
    for (int h = 0; h < nc; h++) {
        int ph = perm[h];
        json probs = json::array();
        for (int a = 0; a < nact; a++) {
            size_t idx = base + (size_t)a * nc + ph;
            float v = (idx < av.size()) ? av[idx] : 0.0f;
            probs.push_back(v);
        }
        jstrat[sg.ranges[nd.player][h].label] = probs;
    }
    return jstrat;
}

// Full 2-level iso variant (flop): the trainable holds only ragged river reps. For a
// full (turn tf, river rf) runout, map onto the rep slot by composing the two suit
// swaps: the turn swap takes the river to rfp = iso_riverperm[tf][rf] in the turn
// rep's frame and relabels hands by iso_perm[tf]; within that frame the river swap
// takes rfp to its river rep slot (riv_fullslot[t][rfp]) and relabels by riv_perm.
static json strat_map_for_flop2_deal(const Subgame& sg, const Node& nd,
                                     const std::vector<float>& av,
                                     int turn_full, int river_full, int ND, int nact, int nc) {
    int p = nd.player;
    int t = sg.iso_rep_slot[turn_full];
    int rfp = sg.iso_riverperm[(size_t)turn_full * ND + river_full];
    int slot = sg.riv_fullslot[(size_t)t * ND + rfp];
    const int* tperm = sg.iso_perm[p].data() + (size_t)turn_full * nc;
    const int* rperm = sg.riv_perm[p].data() + ((size_t)t * ND + rfp) * nc;
    size_t base = (size_t)slot * nact * nc;
    json jstrat = json::object();
    for (int h = 0; h < nc; h++) {
        int ph = rperm[tperm[h]];
        json probs = json::array();
        for (int a = 0; a < nact; a++) {
            size_t idx = base + (size_t)a * nc + ph;
            float v = (idx < av.size()) ? av[idx] : 0.0f;
            probs.push_back(v);
        }
        jstrat[sg.ranges[p][h].label] = probs;
    }
    return jstrat;
}

void dump_strategy_json(const Subgame& sg,
                        const std::vector<std::vector<float>>& avgs,
                        const std::string& path) {
    json root;

    json jboard = json::array();
    for (int c : sg.board) jboard.push_back(c);
    root["board"] = jboard;
    root["root"] = sg.root;

    for (int p = 0; p < 2; p++) {
        json jr = json::array();
        for (const auto& cb : sg.ranges[p]) jr.push_back(cb.label);
        root["range"][p] = jr;
    }

    const int ND = sg.ndeals();
    const int root_round = 3 - sg.chance_levels;   // river=3, turn=2, flop=1

    json jnodes = json::object();
    for (size_t i = 0; i < sg.nodes.size(); i++) {
        const Node& nd = sg.nodes[i];
        if (nd.type != NT_ACTION) continue;

        int nact = (int)nd.labels.size();
        int nc = sg.ncombos(nd.player);
        const std::vector<float>& av = avgs[i];

        json jn;
        jn["player"] = nd.player;
        jn["round"] = nd.round;
        jn["pot"] = nd.pot;
        jn["actions"] = nd.labels;

        // chance levels dealt above this node (0 => no chance, single slot).
        int level = nd.round - root_round;
        if (level <= 0) {
            // No chance above: a single strategy slot (slot 0), flat layout.
            jn["strategy"] = strat_map_for_slot(sg, nd, av, 0, nact, nc);
        } else if (sg.iso_on && level == 1) {
            // Level-1 iso (turn subgame's river nodes, or a flop's turn nodes): the
            // trainable holds only representatives; emit a strategy per real runout by
            // relabeling the rep's hands.
            json jdeals = json::object();
            for (int c = 0; c < sg.iso_nd_full; c++)
                jdeals[sg.iso_labels[c]] = strat_map_for_full_deal(sg, nd, av, c, nact, nc);
            jn["deals"] = jdeals;
        } else if (sg.riv_iso_on && level == 2) {
            // Flop full-2 iso river nodes: the trainable holds ragged river reps; emit
            // a strategy per full (turn, river) runout by composing the turn and river
            // suit swaps onto the rep slot. Key is "turn,river".
            json jdeals = json::object();
            for (int c = 0; c < sg.iso_nd_full; c++) {
                for (int r = 0; r < ND; r++) {
                    if (sg.iso_labels[c] == sg.deal_strs[r]) continue;   // impossible: turn==river
                    std::string key = sg.iso_labels[c] + "," + sg.deal_strs[r];
                    jdeals[key] = strat_map_for_flop2_deal(sg, nd, av, c, r, ND, nact, nc);
                }
            }
            jn["deals"] = jdeals;
        } else if (sg.iso_on && level == 2) {
            // Flop iso river nodes: expand both the reduced turn (via perm) and the
            // full river (via the river-index remap). Key is "turn,river".
            const int RD = ND;   // river is the full deal set
            json jdeals = json::object();
            for (int c = 0; c < sg.iso_nd_full; c++) {
                for (int r = 0; r < RD; r++) {
                    if (sg.iso_labels[c] == sg.deal_strs[r]) continue;   // impossible: turn==river
                    std::string key = sg.iso_labels[c] + "," + sg.deal_strs[r];
                    jdeals[key] = strat_map_for_flop_deal(sg, nd, av, c, r, RD, nact, nc);
                }
            }
            jn["deals"] = jdeals;
        } else {
            // One trainset per compound runout. Slot b is base-ND with `level`
            // digits, most-significant = first dealt (turn before river). Emit a
            // strategy per valid runout keyed by the comma-joined card labels so
            // the CPU loader can map it back to getTrainable(deal).
            long long nsets = 1; for (int L = 0; L < level; L++) nsets *= ND;
            json jdeals = json::object();
            for (long long b = 0; b < nsets; b++) {
                // decode digits (most significant first) and skip impossible
                // runouts that deal the same card twice.
                std::vector<int> didx(level);
                long long rem = b;
                bool ok = true;
                for (int k = level - 1; k >= 0; k--) { didx[k] = (int)(rem % ND); rem /= ND; }
                for (int x = 0; x < level && ok; x++)
                    for (int y = x + 1; y < level && ok; y++)
                        if (didx[x] == didx[y]) ok = false;
                if (!ok) continue;
                std::string key;
                for (int k = 0; k < level; k++) {
                    if (k) key += ",";
                    key += sg.deal_strs[didx[k]];
                }
                jdeals[key] = strat_map_for_slot(sg, nd, av, (int)b, nact, nc);
            }
            jn["deals"] = jdeals;
        }
        jnodes[std::to_string(i)] = jn;
    }
    root["nodes"] = jnodes;

    std::ofstream f(path);
    if (!f) throw std::runtime_error("cannot open dump file: " + path);
    f << root.dump(1);
}

} // namespace texgpu
