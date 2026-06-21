#include "subgame.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace texgpu {

static uint64_t card_int_to_long(int c) {
    // Mirror of Card::boardInt2long: bit position = card int.
    return (uint64_t)1 << c;
}

Subgame load_subgame(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open subgame file: " + path);

    Subgame sg;
    std::string line;

    // Nodes may appear before fully sized; we index by id, so pre-size after "nnodes".
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string tag;
        ss >> tag;

        if (tag == "board") {
            int n; ss >> n;
            sg.board.resize(n);
            sg.board_long = 0;
            for (int i = 0; i < n; i++) {
                ss >> sg.board[i];
                sg.board_long |= card_int_to_long(sg.board[i]);
            }
        } else if (tag == "range") {
            int player, n; ss >> player >> n;
            sg.ranges[player].resize(n);
            for (int i = 0; i < n; i++) {
                std::getline(in, line);
                std::istringstream rs(line);
                Combo c;
                rs >> c.card1 >> c.card2 >> c.weight >> c.rank >> c.label;
                sg.ranges[player][i] = c;
            }
        } else if (tag == "chancelevels") {
            int nd; ss >> sg.chance_levels >> nd;
            sg.has_chance = true;
        } else if (tag == "pdeals") {
            sg.possible_deals.resize(sg.chance_levels);
            for (int i = 0; i < sg.chance_levels; i++) ss >> sg.possible_deals[i];
        } else if (tag == "deals") {
            int n; ss >> n;
            sg.has_chance = true;
            sg.deal_cards.resize(n);
            std::getline(in, line);
            std::istringstream ds(line);
            for (int i = 0; i < n; i++) ds >> sg.deal_cards[i];
            sg.deal_strs.resize(n);
            std::getline(in, line);
            std::istringstream ss2(line);
            for (int i = 0; i < n; i++) ss2 >> sg.deal_strs[i];
        } else if (tag == "dealranks") {
            int player, ndeals, nc; ss >> player >> ndeals >> nc;
            sg.dealrank[player].resize((size_t)ndeals * nc);
            for (int d = 0; d < ndeals; d++) {
                std::getline(in, line);
                std::istringstream rs(line);
                for (int i = 0; i < nc; i++) rs >> sg.dealrank[player][(size_t)d * nc + i];
            }
        } else if (tag == "turnreps") {
            int n; ss >> n;
            sg.iso_level1_cards.resize(n);
            std::getline(in, line);
            std::istringstream ds(line);
            for (int i = 0; i < n; i++) ds >> sg.iso_level1_cards[i];
            std::getline(in, line);   // labels line (unused; dump keys use iso_labels + deal_strs)
        } else if (tag == "iso") {
            ss >> sg.iso_nd_full;
            sg.iso_on = true;
        } else if (tag == "isorepslot") {
            sg.iso_rep_slot.resize(sg.iso_nd_full);
            for (int i = 0; i < sg.iso_nd_full; i++) ss >> sg.iso_rep_slot[i];
        } else if (tag == "isolabels") {
            sg.iso_labels.resize(sg.iso_nd_full);
            for (int i = 0; i < sg.iso_nd_full; i++) ss >> sg.iso_labels[i];
        } else if (tag == "isoperm") {
            int player, nd, nc; ss >> player >> nd >> nc;
            sg.iso_perm[player].resize((size_t)nd * nc);
            for (int d = 0; d < nd; d++) {
                std::getline(in, line);
                std::istringstream rs(line);
                for (int i = 0; i < nc; i++) rs >> sg.iso_perm[player][(size_t)d * nc + i];
            }
        } else if (tag == "isorivperm") {
            int nd, ndr; ss >> nd >> ndr;
            sg.iso_riverperm.resize((size_t)nd * ndr);
            for (int d = 0; d < nd; d++) {
                std::getline(in, line);
                std::istringstream rs(line);
                for (int i = 0; i < ndr; i++) rs >> sg.iso_riverperm[(size_t)d * ndr + i];
            }
        } else if (tag == "nnodes") {
            int n; ss >> n;
            sg.nodes.resize(n);
        } else if (tag == "chance") {
            int id, round, child; double pot;
            ss >> id >> round >> pot >> child;
            Node& nd = sg.nodes.at(id);
            nd.type = NT_CHANCE;
            nd.round = round;
            nd.pot = pot;
            nd.children.assign(1, child);
        } else if (tag == "root") {
            ss >> sg.root;
        } else if (tag == "action") {
            int id, player, round, nact; double pot;
            ss >> id >> player >> round >> pot >> nact;
            Node& nd = sg.nodes.at(id);
            nd.type = NT_ACTION;
            nd.player = player;
            nd.round = round;
            nd.pot = pot;
            nd.children.resize(nact);
            nd.labels.resize(nact);
            for (int a = 0; a < nact; a++) {
                std::getline(in, line);
                std::istringstream as(line);
                std::string act; int childid;
                as >> act >> childid;          // "act <childid> <label...>"
                std::string label;
                std::getline(as, label);       // rest of line (leading space)
                if (!label.empty() && label[0] == ' ') label.erase(0, 1);
                nd.children[a] = childid;
                nd.labels[a] = label;
            }
        } else if (tag == "showdown") {
            int id, round; double pot, p00, p10, p01, p11;
            ss >> id >> round >> pot >> p00 >> p10 >> p01 >> p11;
            Node& nd = sg.nodes.at(id);
            nd.type = NT_SHOWDOWN;
            nd.round = round;
            nd.pot = pot;
            nd.sd[0][0] = p00; nd.sd[1][0] = p10;
            nd.sd[0][1] = p01; nd.sd[1][1] = p11;
        } else if (tag == "terminal") {
            int id, round; double pot, pay0, pay1;
            ss >> id >> round >> pot >> pay0 >> pay1;
            Node& nd = sg.nodes.at(id);
            nd.type = NT_TERMINAL;
            nd.round = round;
            nd.pot = pot;
            nd.pay[0] = pay0; nd.pay[1] = pay1;
        } else {
            throw std::runtime_error("unknown tag in subgame file: " + tag);
        }
    }
    return sg;
}

} // namespace texgpu
