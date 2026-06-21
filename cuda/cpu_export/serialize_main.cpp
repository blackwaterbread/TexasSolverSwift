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
    parser.parse(argc, argv);

    string input_file = parser.retrieve<string>("input_file");
    string resource_dir = parser.retrieve<string>("resource_dir");
    string out_file = parser.retrieve<string>("out_file");
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
                for (size_t i = 3; i < params.size(); i++) sizes->push_back(stof(params[i]));
            }
        } else if (command == "set_allin_threshold") {
            allin_threshold = stof(paramstr);
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
    if (has_chance) {
        out << "chancelevels " << chance_levels << " " << ND << "\n";
        out << "pdeals";
        for (int v : pdeals) out << " " << v;
        out << "\n";
        out << "deals " << ND << "\n";
        for (int ci : deal_cards) out << ci << " ";
        out << "\n";
        for (int ci : deal_cards) out << Card::intCard2Str(ci) << " ";
        out << "\n";
        // Hand ranks at the completed 5-card board, one row per compound runout.
        // Row index is the base-ND compound deal: turn (1 level) or turn*ND+river
        // (2 levels). rank -1 when the combo collides with a dealt card, or the
        // compound deal repeats a card (impossible runout).
        long long nrows = 1; for (int L = 0; L < chance_levels; L++) nrows *= ND;
        for (int p = 0; p < 2; p++) {
            out << "dealranks " << p << " " << nrows << " " << ranges[p].size() << "\n";
            if (chance_levels == 1) {
                for (int t = 0; t < ND; t++) {
                    int ct = deal_cards[t];
                    uint64_t b5 = board_long | Card::boardInt2long(ct);
                    for (auto& pc : ranges[p]) {
                        int rank = (pc.card1 == ct || pc.card2 == ct)
                                   ? -1 : compairer.get_rank(pc.toBoardLong(), b5);
                        out << rank << " ";
                    }
                    out << "\n";
                }
            } else {  // chance_levels == 2 (flop): rows indexed turn*ND + river
                for (int t = 0; t < ND; t++) {
                    int ct = deal_cards[t];
                    for (int r = 0; r < ND; r++) {
                        int cr = deal_cards[r];
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
