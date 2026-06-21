// Serializes a RIVER subgame (fixed 5-card board, no chance nodes) into a plain
// text file the CUDA engine consumes. Mirrors CommandLineTool's setup so the
// produced tree/ranges match the CPU golden run exactly.
//
// Usage: SerializeRiver -i <config.txt> -r <resource_dir> -o <out_file>
//
// Output format (whitespace-delimited; action labels are the rest of their line):
//   board <n> c0 c1 ... cn-1
//   range <player> <ncombos>
//     <card1> <card2> <weight> <rank> <handstr>        x ncombos
//   nnodes <count>
//   root <id>
//   action <id> <player> <round> <pot> <nactions>
//     act <childid> <label...>                          x nactions
//   showdown <id> <round> <pot> <p00> <p10> <p01> <p11>
//   terminal <id> <round> <pot> <pay0> <pay1>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <memory>
#include <functional>
#include <algorithm>

#include "include/library.h"
#include "include/Card.h"
#include "include/Deck.h"
#include "include/GameTree.h"
#include "include/compairer/Dic5Compairer.h"
#include "include/tools/PrivateRangeConverter.h"
#include "include/tools/GameTreeBuildingSettings.h"
#include "include/tools/StreetSetting.h"
#include "include/tools/argparse.hpp"
#include "include/solver/PCfrSolver.h"
#include "include/nodes/ActionNode.h"
#include "include/nodes/ShowdownNode.h"
#include "include/nodes/TerminalNode.h"
#include "include/nodes/ChanceNode.h"
#include "include/Deck.h"
#include "include/ranges/PrivateCards.h"

using namespace std;

static StreetSetting empty_setting() {
    return StreetSetting(vector<float>{}, vector<float>{}, vector<float>{}, true);
}

// Parses one bet-size token. "x" suffix means a multiple of the pot expressed in
// hundredths (matches the GUI's sizes_convert): "2.5x" -> 250, "50" -> 50. The
// in-memory CPU tree the GUI builds uses the same convention, so the serialized
// tree must too or node ids misalign on injection.
static float parse_bet_size(const string& tok) {
    if (!tok.empty() && (tok.back() == 'x' || tok.back() == 'X'))
        return stof(tok.substr(0, tok.size() - 1)) * 100.0f;
    return stof(tok);
}

// Same filtering as the solver's noDuplicateRange: drop combos that collide with
// the board. (Duplicate detection is unnecessary for our controlled ranges.)
static vector<PrivateCards> filterRange(const vector<PrivateCards>& range, uint64_t board_long) {
    vector<PrivateCards> out;
    for (const PrivateCards& pc : range) {
        uint64_t hand_long = Card::boardInts2long(const_cast<PrivateCards&>(pc).get_hands());
        if (!Card::boardsHasIntercept(hand_long, board_long)) out.push_back(pc);
    }
    return out;
}

int main(int argc, const char** argv) {
    ArgumentParser parser;
    parser.addArgument("-i", "--input_file", 1, true);
    parser.addArgument("-r", "--resource_dir", 1, true);
    parser.addArgument("-o", "--out_file", 1, true);
    parser.addArgument("--no_iso", 1, true);          // "1" disables suit isomorphism (full deals; for A/B benchmarking)
    parser.parse(argc, argv);

    string input_file = parser.retrieve<string>("input_file");
    string resource_dir = parser.retrieve<string>("resource_dir");
    string out_file = parser.retrieve<string>("out_file");
    bool no_iso = parser.retrieve<string>("no_iso") == "1";
    if (resource_dir.empty()) resource_dir = "./resources";
    if (out_file.empty()) out_file = "subgame.txt";

    // ---- compairer (for hand ranks) + deck ----
    string compairer_file = resource_dir + "/compairer/card5_dic_sorted.txt";
    string compairer_bin = resource_dir + "/compairer/card5_dic_zipped.bin";
    Dic5Compairer compairer(compairer_file, 2598961, compairer_bin);

    vector<string> ranks_vector = string_split(string("2,3,4,5,6,7,8,9,T,J,Q,K,A"), ',');
    vector<string> suits_vector = string_split(string("c,d,h,s"), ',');
    Deck deck(ranks_vector, suits_vector);

    // ---- defaults mirror CommandLineTool ----
    float ip_commit = 5, oop_commit = 5;
    float stack = 25;
    int current_round = 3;
    int raise_limit = 4;
    float small_blind = 0.5f, big_blind = 1;
    float allin_threshold = 0.67f;
    string range_ip, range_oop, board_str;

    StreetSetting flop_ip = empty_setting(), turn_ip = empty_setting(), river_ip = empty_setting();
    StreetSetting flop_oop = empty_setting(), turn_oop = empty_setting(), river_oop = empty_setting();
    auto gtbs = make_shared<GameTreeBuildingSettings>(flop_ip, turn_ip, river_ip, flop_oop, turn_oop, river_oop);

    shared_ptr<GameTree> game_tree;

    // ---- parse config ----
    ifstream infile(input_file);
    string line;
    while (getline(infile, line)) {
        if (line.empty()) continue;
        vector<string> toks;
        // split on first space only: command + paramstr
        size_t sp = line.find(' ');
        string command = sp == string::npos ? line : line.substr(0, sp);
        string paramstr = sp == string::npos ? "" : line.substr(sp + 1);

        if (command == "set_pot") {
            ip_commit = stof(paramstr) / 2; oop_commit = stof(paramstr) / 2;
        } else if (command == "set_effective_stack") {
            stack = stof(paramstr) + ip_commit;
        } else if (command == "set_board") {
            board_str = paramstr;
            vector<string> b = string_split(paramstr, ',');
            if (b.size() == 3) current_round = 1;
            else if (b.size() == 4) current_round = 2;
            else if (b.size() == 5) current_round = 3;
            else throw runtime_error("board not recognized");
        } else if (command == "set_range_ip") {
            range_ip = paramstr;
        } else if (command == "set_range_oop") {
            range_oop = paramstr;
        } else if (command == "set_bet_sizes") {
            vector<string> params = string_split(paramstr, ',');
            if (params.size() < 3) throw runtime_error("param number error");
            StreetSetting& s = gtbs->get_setting(params[0], params[1]);
            string bet_type = params[2];
            vector<float>* sizes = nullptr;
            if (bet_type == "allin") s.allin = true;
            else if (bet_type == "bet") sizes = &s.bet_sizes;
            else if (bet_type == "raise") sizes = &s.raise_sizes;
            else if (bet_type == "donk") sizes = &s.donk_sizes;
            else throw runtime_error("bad bet type");
            if (sizes) {
                sizes->clear();
                for (size_t i = 3; i < params.size(); i++) sizes->push_back(parse_bet_size(params[i]));
            }
        } else if (command == "set_allin_threshold") {
            allin_threshold = stof(paramstr);
        } else if (command == "set_raise_limit") {
            raise_limit = stoi(paramstr);
        } else if (command == "build_tree") {
            game_tree = make_shared<GameTree>(deck, oop_commit, ip_commit, current_round,
                                              raise_limit, small_blind, big_blind, stack,
                                              *gtbs.get(), allin_threshold);
        }
        // ignore solve/iteration/dump commands
    }

    if (game_tree == nullptr) {
        // build with whatever we parsed even if build_tree wasn't present
        game_tree = make_shared<GameTree>(deck, oop_commit, ip_commit, current_round,
                                          raise_limit, small_blind, big_blind, stack,
                                          *gtbs.get(), allin_threshold);
    }

    // ---- board ints + long ----
    vector<string> board_arr = string_split(board_str, ',');
    vector<int> board_ints;
    for (const string& s : board_arr) board_ints.push_back(Card::strCard2int(s));
    uint64_t board_long = Card::boardInts2long(board_ints);

    // ---- ranges (same filtering as the solver) ----
    vector<PrivateCards> range0 = PrivateRangeConverter::rangeStr2Cards(range_ip, board_ints);
    vector<PrivateCards> range1 = PrivateRangeConverter::rangeStr2Cards(range_oop, board_ints);
    range0 = filterRange(range0, board_long);
    range1 = filterRange(range1, board_long);
    vector<vector<PrivateCards>> ranges = {range0, range1};

    // ---- flatten tree (preorder ids) ----
    vector<shared_ptr<GameTreeNode>> nodes;
    unordered_map<GameTreeNode*, int> id;
    function<void(shared_ptr<GameTreeNode>)> assign = [&](shared_ptr<GameTreeNode> n) {
        id[n.get()] = (int)nodes.size();
        nodes.push_back(n);
        if (n->getType() == GameTreeNode::GameTreeNodeType::ACTION) {
            auto an = dynamic_pointer_cast<ActionNode>(n);
            for (auto& c : an->getChildrens()) assign(c);
        } else if (n->getType() == GameTreeNode::GameTreeNodeType::CHANCE) {
            auto cn = dynamic_pointer_cast<ChanceNode>(n);
            assign(cn->getChildren());
        }
    };
    assign(game_tree->getRoot());

    // ---- deals (valid runout cards) + chance levels ----
    // One chance level per street still to come: flop=2 (turn,river), turn=1
    // (river), river=0. The same deal-card set (deck minus the current board) is
    // dealt at every level; deeper levels mask the card a shallower one used.
    vector<int> deal_cards;            // deck card ints not on board
    for (const Card& c : deck.getCards()) {
        int ci = const_cast<Card&>(c).getCardInt();
        if (!Card::boardsHasIntercept(Card::boardInt2long(ci), board_long)) deal_cards.push_back(ci);
    }
    int ND = (int)deal_cards.size();
    int deck_n = (int)deck.getCards().size();
    int board_n = (int)board_ints.size();
    int chance_levels = 3 - current_round;          // flop=2, turn=1, river=0
    bool has_chance = chance_levels > 0;
    // reach divisor for the chance dealing the (L+1)-th card (board grown by L).
    vector<int> pdeals;
    for (int L = 0; L < chance_levels; L++) pdeals.push_back(deck_n - board_n - L - 2);

    // ---- suit isomorphism (level-1 runout card) ----
    // Two suits are equivalent iff the (root) board carries the same rank-set in both
    // AND both players' ranges are invariant under swapping them. Equivalent level-1
    // runout cards (same rank, equivalent suits) collapse to one representative; the
    // GPU engine trains only representatives and re-expands at the level-1 chance
    // reduce via the suit permutation. Card int = rank*4 + suit (ci%4 = suit). The
    // reduced card is the river for a turn subgame and the turn for a flop subgame;
    // either way it is the FIRST card dealt off the root board, so canon/perm are
    // computed identically. For the flop the second level (river) stays full.
    bool iso_on = false;
    int nd_iso = ND;
    vector<int> reps;                       // representative card ints (size nd_iso)
    vector<int> rep_slot_full(ND, 0);       // [ND] representative slot per full deal
    vector<vector<int>> iso_perm[2];        // [player][full deal] -> permuted hand idx
    vector<vector<int>> iso_rivperm;        // flop only: [full turn] -> river index swap
    // Full 2-level iso (flop only): river iso is recomputed inside each turn rep's
    // frame (board = flop + turn-rep card), giving a RAGGED set of river reps per
    // turn rep. Everything below is in turn-rep frame, indexed by deal_cards (valid
    // when deal_cards[r] != turn-rep card).
    bool riv_iso_on = false;
    vector<int> riv_off;                     // [NT+1] prefix offsets into the flat river-rep dim
    vector<int> riv_rep_cards;               // [total] river rep card int per absolute slot
    vector<int> riv_slot2turn;               // [total] turn rep index per absolute slot
    vector<vector<int>> riv_fullslot;        // [NT][ND] abs river-rep slot (-1 if r==turn card)
    vector<int> riv_perm_flat[2];            // [NT*ND*nc] full-river hand -> river-rep hand
    if (chance_levels >= 1) {
        int color_hash[4] = {0, 0, 0, 0};   // ranks present on board per suit
        for (int bc : board_ints) color_hash[bc % 4] |= (1 << (bc / 4));
        auto sorted_key = [](int c1, int c2) { if (c1 > c2) std::swap(c1, c2); return c1 * 52 + c2; };
        auto range_invariant = [&](int a, int b, const vector<PrivateCards>& rg) {
            unordered_map<int, float> wmap;
            for (auto& pc : rg) wmap[sorted_key(pc.card1, pc.card2)] = pc.weight;
            auto sw = [&](int x) { return x % 4 == a ? x - a + b : (x % 4 == b ? x - b + a : x); };
            for (auto& pc : rg) {
                auto it = wmap.find(sorted_key(sw(pc.card1), sw(pc.card2)));
                if (it == wmap.end() || it->second != pc.weight) return false;
            }
            return true;
        };
        // union-find over the 4 suits; merge equivalent pairs (skipped under --no_iso,
        // leaving every suit its own class so the full deal set is emitted unchanged).
        int uf[4] = {0, 1, 2, 3};
        function<int(int)> find = [&](int x) { return uf[x] == x ? x : uf[x] = find(uf[x]); };
        if (!no_iso)
            for (int a = 0; a < 4; a++)
                for (int b = a + 1; b < 4; b++)
                    if (color_hash[a] == color_hash[b] &&
                        range_invariant(a, b, range0) && range_invariant(a, b, range1)) {
                        int ra = find(a), rb = find(b);
                        uf[ra > rb ? ra : rb] = (ra < rb ? ra : rb);
                    }
        int canon[4];                       // smallest suit index in each class
        for (int s = 0; s < 4; s++) { canon[s] = s; for (int j = 0; j < s; j++) if (find(j) == find(s)) { canon[s] = canon[j]; break; } }

        unordered_map<int, int> repcard_slot, cardint_fullidx;
        for (int t = 0; t < ND; t++) cardint_fullidx[deal_cards[t]] = t;
        for (int t = 0; t < ND; t++) {
            int c = deal_cards[t], s = c % 4;
            if (canon[s] == s) { repcard_slot[c] = (int)reps.size(); reps.push_back(c); }
        }
        nd_iso = (int)reps.size();
        for (int t = 0; t < ND; t++) {
            int c = deal_cards[t], s = c % 4;
            rep_slot_full[t] = repcard_slot[c - s + canon[s]];
        }
        // per-player hand permutation for each full deal (transposition s<->canon[s]).
        for (int p = 0; p < 2; p++) {
            unordered_map<int, int> pidx;
            for (int i = 0; i < (int)ranges[p].size(); i++) pidx[sorted_key(ranges[p][i].card1, ranges[p][i].card2)] = i;
            iso_perm[p].resize(ND);
            for (int t = 0; t < ND; t++) {
                int s = deal_cards[t] % 4, a = std::min(s, canon[s]), b = std::max(s, canon[s]);
                auto sw = [&](int x) { return x % 4 == a ? x - a + b : (x % 4 == b ? x - b + a : x); };
                vector<int>& pv = iso_perm[p][t]; pv.resize(ranges[p].size());
                for (int i = 0; i < (int)ranges[p].size(); i++) {
                    auto it = pidx.find(sorted_key(sw(ranges[p][i].card1), sw(ranges[p][i].card2)));
                    pv[i] = (it != pidx.end()) ? it->second : i;
                }
            }
        }
        iso_on = (nd_iso < ND);

        // Flop: river index permutation under each full turn card's suit swap (for the
        // dump, which keeps the river full while reducing the turn). Row t, col r ->
        // index of swap_t(deal_cards[r]) in deal_cards.
        if (iso_on && chance_levels == 2) {
            iso_rivperm.resize(ND);
            for (int t = 0; t < ND; t++) {
                int s = deal_cards[t] % 4, a = std::min(s, canon[s]), b = std::max(s, canon[s]);
                auto sw = [&](int x) { return x % 4 == a ? x - a + b : (x % 4 == b ? x - b + a : x); };
                iso_rivperm[t].resize(ND);
                for (int r = 0; r < ND; r++) iso_rivperm[t][r] = cardint_fullidx[sw(deal_cards[r])];
            }
        }

        // Full 2-level: river iso inside each turn rep's frame. For turn rep t (card
        // ct = reps[t]) recompute the suit classes of board+ct, collapse equivalent
        // river cards to representatives, and record a per-(turn rep, full river) rep
        // slot + hand permutation. The result is ragged (river-rep count varies by
        // turn rep: a turn card that breaks a flop suit symmetry leaves the river with
        // fewer/no equivalences). Indexed by deal_cards in turn-rep frame (the river
        // dealt below turn rep t ranges over deck-flop-ct = deal_cards minus {ct}).
        if (iso_on && chance_levels == 2) {
            int NT = nd_iso;
            // per-player hand index map (sorted_key -> combo index), reused per turn rep.
            unordered_map<int, int> pidx[2];
            for (int p = 0; p < 2; p++)
                for (int i = 0; i < (int)ranges[p].size(); i++)
                    pidx[p][sorted_key(ranges[p][i].card1, ranges[p][i].card2)] = i;
            riv_off.assign(NT + 1, 0);
            riv_fullslot.assign(NT, vector<int>(ND, -1));
            for (int p = 0; p < 2; p++) riv_perm_flat[p].assign((size_t)NT * ND * ranges[p].size(), 0);
            int counter = 0;
            for (int t = 0; t < NT; t++) {
                int ct = reps[t];
                int ch2[4]; for (int s = 0; s < 4; s++) ch2[s] = color_hash[s];
                ch2[ct % 4] |= (1 << (ct / 4));
                int uf2[4] = {0, 1, 2, 3};
                function<int(int)> find2 = [&](int x) { return uf2[x] == x ? x : uf2[x] = find2(uf2[x]); };
                if (!no_iso)
                    for (int a = 0; a < 4; a++)
                        for (int b = a + 1; b < 4; b++)
                            if (ch2[a] == ch2[b] && range_invariant(a, b, range0) && range_invariant(a, b, range1)) {
                                int ra = find2(a), rb = find2(b);
                                uf2[ra > rb ? ra : rb] = (ra < rb ? ra : rb);
                            }
                int canon2[4];
                for (int s = 0; s < 4; s++) { canon2[s] = s; for (int j = 0; j < s; j++) if (find2(j) == find2(s)) { canon2[s] = canon2[j]; break; } }
                // assign absolute slots for this turn rep's river representatives
                riv_off[t] = counter;
                unordered_map<int, int> repcard_slot2;     // river rep card int -> abs slot
                for (int r = 0; r < ND; r++) {
                    int cr = deal_cards[r]; if (cr == ct) continue;
                    int s = cr % 4;
                    if (canon2[s] == s) { repcard_slot2[cr] = counter; riv_rep_cards.push_back(cr); riv_slot2turn.push_back(t); counter++; }
                }
                // full-river -> rep slot + per-player hand perm (suit swap s<->canon2[s])
                for (int r = 0; r < ND; r++) {
                    int cr = deal_cards[r]; if (cr == ct) { riv_fullslot[t][r] = -1; continue; }
                    int s = cr % 4, a = std::min(s, canon2[s]), b = std::max(s, canon2[s]);
                    auto sw = [&](int x) { return x % 4 == a ? x - a + b : (x % 4 == b ? x - b + a : x); };
                    riv_fullslot[t][r] = repcard_slot2[cr - s + canon2[s]];
                    for (int p = 0; p < 2; p++) {
                        size_t base = ((size_t)t * ND + r) * ranges[p].size();
                        for (int i = 0; i < (int)ranges[p].size(); i++) {
                            auto it = pidx[p].find(sorted_key(sw(ranges[p][i].card1), sw(ranges[p][i].card2)));
                            riv_perm_flat[p][base + i] = (it != pidx[p].end()) ? it->second : i;
                        }
                    }
                }
            }
            riv_off[NT] = counter;
            riv_iso_on = true;

            // Composed self-check (equilibrium-independent ground truth): for every full
            // (turn tf, river rf) the engine/dump maps hand h to the rep trainable via
            // turn perm then river perm; the completed-board ranks must match. rfp is the
            // river index in turn rep t's frame (after the turn suit swap, == iso_rivperm).
            for (int p = 0; p < 2; p++) {
                int nc = (int)ranges[p].size();
                for (int tf = 0; tf < ND; tf++) {
                    int ct = deal_cards[tf], t = rep_slot_full[tf], ctrep = reps[t];
                    for (int rf = 0; rf < ND; rf++) {
                        int cr = deal_cards[rf]; if (cr == ct) continue;
                        int rfp = iso_rivperm[tf][rf];
                        int slot = riv_fullslot[t][rfp];
                        int crrep = riv_rep_cards[slot];
                        uint64_t b5 = board_long | Card::boardInt2long(ct) | Card::boardInt2long(cr);
                        uint64_t b5r = board_long | Card::boardInt2long(ctrep) | Card::boardInt2long(crrep);
                        for (int h = 0; h < nc; h++) {
                            PrivateCards& pc = ranges[p][h];
                            int rk_c = (pc.card1 == ct || pc.card2 == ct || pc.card1 == cr || pc.card2 == cr)
                                       ? -1 : compairer.get_rank(pc.toBoardLong(), b5);
                            int th = iso_perm[p][tf][h];
                            int ph = riv_perm_flat[p][((size_t)t * ND + rfp) * nc + th];
                            PrivateCards& pr = ranges[p][ph];
                            int rk_r = (pr.card1 == ctrep || pr.card2 == ctrep || pr.card1 == crrep || pr.card2 == crrep)
                                       ? -1 : compairer.get_rank(pr.toBoardLong(), b5r);
                            if (rk_c != rk_r)
                                throw runtime_error("flop full-2 iso self-check failed: player " + to_string(p) +
                                    " turn " + to_string(tf) + " river " + to_string(rf) + " hand " + to_string(h) +
                                    " (" + to_string(rk_c) + " vs " + to_string(rk_r) + ")");
                        }
                    }
                }
            }
            cout << "flop full-2 iso self-check passed: turn " << ND << "->" << NT
                 << " reps, river reps total " << counter << " (vs full " << NT * ND << ")\n";
        }

        // Self-check (equilibrium-independent): hand strengths are suit-permutation
        // invariant, so the completed-board rank at runout c must equal the rank at
        // its representative with the hands relabeled by the suit swap. Validates
        // canon + perm before any solve. The turn subgame completes the board with c
        // alone; the flop subgame still needs a river, and the swap must carry to it
        // too (rank(h@flop+c+r) == rank(perm(h)@flop+rep+swap(r))).
        if (iso_on && chance_levels == 1) {
            for (int p = 0; p < 2; p++) {
                for (int t = 0; t < ND; t++) {
                    int ct = deal_cards[t];
                    uint64_t bt = board_long | Card::boardInt2long(ct);
                    int crep = reps[rep_slot_full[t]];
                    uint64_t brep = board_long | Card::boardInt2long(crep);
                    for (int h = 0; h < (int)ranges[p].size(); h++) {
                        PrivateCards& pc = ranges[p][h];
                        int rk_c = (pc.card1 == ct || pc.card2 == ct) ? -1 : compairer.get_rank(pc.toBoardLong(), bt);
                        PrivateCards& pr = ranges[p][iso_perm[p][t][h]];
                        int rk_r = (pr.card1 == crep || pr.card2 == crep) ? -1 : compairer.get_rank(pr.toBoardLong(), brep);
                        if (rk_c != rk_r)
                            throw runtime_error("iso self-check failed: player " + to_string(p) + " deal " + to_string(t) +
                                                " hand " + to_string(h) + " (" + to_string(rk_c) + " vs " + to_string(rk_r) + ")");
                    }
                }
            }
            cout << "iso self-check passed: ND " << ND << " -> " << nd_iso << " representatives\n";
        } else if (iso_on && chance_levels == 2) {
            for (int p = 0; p < 2; p++) {
                for (int t = 0; t < ND; t++) {
                    int ct = deal_cards[t], s = ct % 4, a = std::min(s, canon[s]), b = std::max(s, canon[s]);
                    auto sw = [&](int x) { return x % 4 == a ? x - a + b : (x % 4 == b ? x - b + a : x); };
                    int crep = reps[rep_slot_full[t]];
                    for (int rr = 0; rr < ND; rr++) {
                        int cr = deal_cards[rr];
                        if (cr == ct) continue;
                        int crsw = sw(cr);
                        uint64_t b5 = board_long | Card::boardInt2long(ct) | Card::boardInt2long(cr);
                        uint64_t b5r = board_long | Card::boardInt2long(crep) | Card::boardInt2long(crsw);
                        for (int h = 0; h < (int)ranges[p].size(); h++) {
                            PrivateCards& pc = ranges[p][h];
                            int rk_c = (pc.card1 == ct || pc.card2 == ct || pc.card1 == cr || pc.card2 == cr)
                                       ? -1 : compairer.get_rank(pc.toBoardLong(), b5);
                            PrivateCards& pr = ranges[p][iso_perm[p][t][h]];
                            int rk_r = (pr.card1 == crep || pr.card2 == crep || pr.card1 == crsw || pr.card2 == crsw)
                                       ? -1 : compairer.get_rank(pr.toBoardLong(), b5r);
                            if (rk_c != rk_r)
                                throw runtime_error("flop iso self-check failed: player " + to_string(p) + " turn " + to_string(t) +
                                                    " river " + to_string(rr) + " hand " + to_string(h) +
                                                    " (" + to_string(rk_c) + " vs " + to_string(rk_r) + ")");
                        }
                    }
                }
            }
            cout << "flop iso self-check passed: turn ND " << ND << " -> " << nd_iso << " representatives (river full)\n";
        }
    }

    // ---- write ----
    ofstream out(out_file);
    out.precision(9);

    out << "board " << board_ints.size();
    for (int c : board_ints) out << " " << c;
    out << "\n";

    bool board_is_river = ((int)board_ints.size() == 5);
    for (int p = 0; p < 2; p++) {
        out << "range " << p << " " << ranges[p].size() << "\n";
        for (auto& pc : ranges[p]) {
            // base rank only meaningful at a 5-card (river) board; otherwise unused
            int rank = board_is_river ? compairer.get_rank(pc.toBoardLong(), board_long) : 0;
            out << pc.card1 << " " << pc.card2 << " " << pc.weight << " " << rank
                << " " << pc.toString() << "\n";
        }
    }

    // ---- chance data (turn/flop subgames; river dealt by chance node(s)) ----
    // For the turn (1 level) the engine deals only the iso representatives (nd_iso);
    // the chance reduce re-expands to all ND cards via the iso block below. For the
    // flop (2 levels) iso is not applied yet, so reps == all deal cards (nd_iso==ND).
    if (has_chance) {
        // `deals` is the DEEPEST level's card set (the one dealrank/showdown index
        // directly): the reduced river reps for a turn subgame, the full river for a
        // flop (whose shallower turn level is reduced separately via `turnreps`).
        const vector<int>& emit_deals = (chance_levels == 1) ? reps : deal_cards;
        int nd_eff = (int)emit_deals.size();
        out << "chancelevels " << chance_levels << " " << nd_eff << "\n";
        out << "pdeals";
        for (int v : pdeals) out << " " << v;
        out << "\n";
        out << "deals " << nd_eff << "\n";
        for (int ci : emit_deals) out << ci << " ";
        out << "\n";
        for (int ci : emit_deals) out << Card::intCard2Str(ci) << " ";
        out << "\n";
        // Flop: the shallower (turn) level deals only the representatives. The river
        // (deepest, `deals` above) stays full; the engine re-expands the turn at the
        // level-1 chance reduce. Absent for the turn subgame (turn==deepest there).
        if (chance_levels == 2) {
            out << "turnreps " << reps.size() << "\n";
            for (int ci : reps) out << ci << " ";
            out << "\n";
            for (int ci : reps) out << Card::intCard2Str(ci) << " ";
            out << "\n";
        }
        // Hand ranks at the completed 5-card board, one row per compound runout. Row
        // index: turn (1 level) = rep slot; flop level-1 iso = turn_rep*ND + river
        // (turn reps, river full); flop full-2 iso = the absolute river-rep slot
        // (board flop + turn-rep card + river-rep card, in riv_slot2turn order). rank
        // -1 when the combo collides with a dealt card or the runout repeats a card.
        long long nrows = (chance_levels == 1) ? (long long)reps.size()
                        : riv_iso_on            ? (long long)riv_off[(int)reps.size()]
                                                : (long long)reps.size() * ND;
        for (int p = 0; p < 2; p++) {
            out << "dealranks " << p << " " << nrows << " " << ranges[p].size() << "\n";
            if (chance_levels == 1) {
                for (int ct : reps) {
                    uint64_t b5 = board_long | Card::boardInt2long(ct);
                    for (auto& pc : ranges[p]) {
                        int rank = (pc.card1 == ct || pc.card2 == ct)
                                   ? -1 : compairer.get_rank(pc.toBoardLong(), b5);
                        out << rank << " ";
                    }
                    out << "\n";
                }
            } else if (riv_iso_on) {  // flop full-2: one row per absolute river-rep slot
                for (long long slot = 0; slot < nrows; slot++) {
                    int ct = reps[riv_slot2turn[slot]], cr = riv_rep_cards[slot];
                    uint64_t b5 = board_long | Card::boardInt2long(ct) | Card::boardInt2long(cr);
                    for (auto& pc : ranges[p]) {
                        int rank = (pc.card1 == ct || pc.card2 == ct ||
                                    pc.card1 == cr || pc.card2 == cr)
                                   ? -1 : compairer.get_rank(pc.toBoardLong(), b5);
                        out << rank << " ";
                    }
                    out << "\n";
                }
            } else {  // chance_levels == 2 (flop), no iso: turn over reps, river full
                for (int ct : reps) {
                    for (int cr : deal_cards) {
                        if (ct == cr) {  // impossible: same card dealt twice
                            for (size_t k = 0; k < ranges[p].size(); k++) out << "-1 ";
                            out << "\n";
                            continue;
                        }
                        uint64_t b5 = board_long | Card::boardInt2long(ct) | Card::boardInt2long(cr);
                        for (auto& pc : ranges[p]) {
                            int rank = (pc.card1 == ct || pc.card2 == ct ||
                                        pc.card1 == cr || pc.card2 == cr)
                                       ? -1 : compairer.get_rank(pc.toBoardLong(), b5);
                            out << rank << " ";
                        }
                        out << "\n";
                    }
                }
            }
        }

        // iso block: full level-1 runout count (ND), each full level-1 card's
        // representative slot, the full level-1 labels (for dump keys), and the
        // per-player hand permutation mapping each full level-1 card's hands onto its
        // representative's hands. The full level-1 set equals deal_cards (deck minus
        // root board) for both turn and flop. Present only when the level reduces.
        if (iso_on) {
            out << "iso " << ND << "\n";
            out << "isorepslot";
            for (int t = 0; t < ND; t++) out << " " << rep_slot_full[t];
            out << "\n";
            out << "isolabels";
            for (int ci : deal_cards) out << " " << Card::intCard2Str(ci);
            out << "\n";
            for (int p = 0; p < 2; p++) {
                out << "isoperm " << p << " " << ND << " " << ranges[p].size() << "\n";
                for (int t = 0; t < ND; t++) {
                    for (int h = 0; h < (int)ranges[p].size(); h++) out << iso_perm[p][t][h] << " ";
                    out << "\n";
                }
            }
            // Flop dump only: the river index permutation under each full turn card's
            // suit swap. A full turn card c's strategy at river r is the rep's strategy
            // at river swap_c(r) (river stays full/unreduced in the trainable), so the
            // dump remaps the river index by this table. Row c, column r -> index of
            // swap_c(deal_cards[r]) in deal_cards. Identity rows for representatives.
            if (chance_levels == 2) {
                out << "isorivperm " << ND << " " << ND << "\n";
                for (int t = 0; t < ND; t++) {
                    for (int r = 0; r < ND; r++) out << iso_rivperm[t][r] << " ";
                    out << "\n";
                }
            }
        }

        // Full 2-level iso ragged river blocks (flop only). Presence of `rivoff` tells
        // the engine to use the per-turn-rep ragged river iso path: `deals`/dealranks
        // above index the absolute river-rep slot, `rivoff` are the prefix offsets per
        // turn rep, `rivrepcards`/`rivslot2turn` describe each slot, `rivfullslot` maps
        // (turn rep, full river index) -> abs slot (-1 if river==turn card), and
        // `rivperm` is the per-player hand relabel for that full river onto its rep.
        if (riv_iso_on) {
            int NT = (int)reps.size();
            int total = riv_off[NT];
            out << "rivoff " << NT << "\n";
            for (int t = 0; t <= NT; t++) out << riv_off[t] << " ";
            out << "\n";
            out << "rivrepcards " << total << "\n";
            for (int s = 0; s < total; s++) out << riv_rep_cards[s] << " ";
            out << "\n";
            out << "rivslot2turn " << total << "\n";
            for (int s = 0; s < total; s++) out << riv_slot2turn[s] << " ";
            out << "\n";
            out << "rivfullslot " << NT << " " << ND << "\n";
            for (int t = 0; t < NT; t++) {
                for (int r = 0; r < ND; r++) out << riv_fullslot[t][r] << " ";
                out << "\n";
            }
            for (int p = 0; p < 2; p++) {
                out << "rivperm " << p << " " << (long long)NT * ND << " " << ranges[p].size() << "\n";
                for (long long row = 0; row < (long long)NT * ND; row++) {
                    size_t base = (size_t)row * ranges[p].size();
                    for (int i = 0; i < (int)ranges[p].size(); i++) out << riv_perm_flat[p][base + i] << " ";
                    out << "\n";
                }
            }
        }
    }

    out << "nnodes " << nodes.size() << "\n";
    out << "root " << id[game_tree->getRoot().get()] << "\n";

    for (auto& n : nodes) {
        int nid = id[n.get()];
        int round = GameTreeNode::gameRound2int(n->getRound());
        double pot = n->getPot();
        switch (n->getType()) {
            case GameTreeNode::GameTreeNodeType::ACTION: {
                auto an = dynamic_pointer_cast<ActionNode>(n);
                auto& acts = an->getActions();
                auto& kids = an->getChildrens();
                out << "action " << nid << " " << an->getPlayer() << " " << round
                    << " " << pot << " " << acts.size() << "\n";
                for (size_t a = 0; a < acts.size(); a++) {
                    out << "act " << id[kids[a].get()] << " " << acts[a].toString() << "\n";
                }
                break;
            }
            case GameTreeNode::GameTreeNodeType::SHOWDOWN: {
                auto sn = dynamic_pointer_cast<ShowdownNode>(n);
                double p00 = sn->get_payoffs(ShowdownNode::ShowDownResult::NOTTIE, 0, 0);
                double p10 = sn->get_payoffs(ShowdownNode::ShowDownResult::NOTTIE, 1, 0);
                double p01 = sn->get_payoffs(ShowdownNode::ShowDownResult::NOTTIE, 0, 1);
                double p11 = sn->get_payoffs(ShowdownNode::ShowDownResult::NOTTIE, 1, 1);
                out << "showdown " << nid << " " << round << " " << pot << " "
                    << p00 << " " << p10 << " " << p01 << " " << p11 << "\n";
                break;
            }
            case GameTreeNode::GameTreeNodeType::TERMINAL: {
                auto tn = dynamic_pointer_cast<TerminalNode>(n);
                vector<double> pays = tn->get_payoffs();
                out << "terminal " << nid << " " << round << " " << pot << " "
                    << pays[0] << " " << pays[1] << "\n";
                break;
            }
            case GameTreeNode::GameTreeNodeType::CHANCE: {
                auto cn = dynamic_pointer_cast<ChanceNode>(n);
                out << "chance " << nid << " " << round << " " << pot << " "
                    << id[cn->getChildren().get()] << "\n";
                break;
            }
            default:
                throw runtime_error("unexpected node type");
        }
    }
    out.close();
    cout << "serialized " << nodes.size() << " nodes, ranges "
         << ranges[0].size() << "/" << ranges[1].size() << " -> " << out_file << endl;
    return 0;
}
