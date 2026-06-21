#pragma once
#include <string>
#include <vector>
#include <cstdint>

// Host-side representation of a serialized RIVER subgame (see cpu_export/serialize_main.cpp).
namespace texgpu {

struct Combo {
    int card1 = 0;
    int card2 = 0;
    float weight = 0.0f;
    int rank = 0;          // smaller = stronger hand
    std::string label;     // e.g. "AdAc"
};

enum NodeType { NT_ACTION = 0, NT_SHOWDOWN = 1, NT_TERMINAL = 2, NT_CHANCE = 3 };

struct Node {
    NodeType type = NT_ACTION;
    int round = 3;
    double pot = 0.0;

    // ACTION
    int player = 0;                       // acting player (0 or 1)
    std::vector<int> children;            // child node id per action (CHANCE: single child in children[0])
    std::vector<std::string> labels;      // action label per action (matches golden keys)

    // SHOWDOWN: payoff to `player` when `winner` wins. p[winner][player].
    double sd[2][2] = {{0, 0}, {0, 0}};   // sd[0][0],sd[1][0],sd[0][1],sd[1][1]

    // TERMINAL: payoff to each player (fold).
    double pay[2] = {0, 0};
};

struct Subgame {
    std::vector<int> board;               // board card ints (3/4/5)
    uint64_t board_long = 0;
    std::vector<Combo> ranges[2];         // per-player combos (filtered by board)
    std::vector<Node> nodes;
    int root = 0;

    // Chance. One chance level per street left to deal: turn subgame = 1 (deal
    // river), flop subgame = 2 (deal turn then river). The same deal-card set
    // (deck minus the current board) is dealt at every level; deeper levels mask
    // the card already used by a shallower one.
    bool has_chance = false;
    int chance_levels = 0;                // # of chance levels (3 - current_round)
    std::vector<int> deal_cards;          // runout card ints (deck minus board)
    std::vector<std::string> deal_strs;   // runout card labels (e.g. "2c") for golden matching
    std::vector<int> possible_deals;      // reach divisor per chance level (size chance_levels)
    // Hand ranks at the completed 5-card board, indexed by the compound runout.
    // dealrank[player] has ND^chance_levels rows of ncombos; the row index is the
    // base-ND compound deal (turn*ND + river for the flop). rank -1 => the combo
    // collides with a dealt card, or the compound deal repeats a card (impossible).
    std::vector<int> dealrank[2];

    // Suit isomorphism (turn subgame only, single chance level). When on, deal_cards
    // / dealrank above hold only the iso representatives (nd_iso = ndeals()); the
    // chance node deals representatives and the reduce re-expands to all iso_nd_full
    // runouts. iso_rep_slot[c] gives the representative slot (0..nd_iso-1) for full
    // deal c, and iso_perm[player][c*nc + h] maps full deal c's hand h onto the
    // representative's hand (the suit-swap permutation). iso_labels[c] is the full
    // runout's card label, used to key the expanded dump. Empty when off.
    bool iso_on = false;
    int iso_nd_full = 0;
    std::vector<int> iso_rep_slot;            // [nd_full]
    std::vector<std::string> iso_labels;      // [nd_full]
    std::vector<int> iso_perm[2];             // [nd_full * ncombos(player)]
    // Flop only: the reduced level-1 (turn) deal cards. The deepest level (river)
    // stays full in deal_cards; here the turn level holds just the representatives.
    // Empty for the turn subgame (its reduced level is the deepest, already in deal_cards).
    std::vector<int> iso_level1_cards;
    // Flop dump only: [nd_full * ND] river-index permutation under each full turn
    // card's suit swap (row = full turn card, col = river index -> remapped index).
    std::vector<int> iso_riverperm;

    int ncombos(int player) const { return (int)ranges[player].size(); }
    int ndeals() const { return (int)deal_cards.size(); }
};

// Parse a serialized subgame file. Throws std::runtime_error on malformed input.
Subgame load_subgame(const std::string& path);

} // namespace texgpu
