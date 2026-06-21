#include "compare_golden.h"

#include <fstream>
#include <cmath>
#include <cstdio>
#include "json.hpp"

using json = nlohmann::json;

namespace texgpu {

struct Stats {
    long count = 0;
    double sum_abs = 0.0;
    double max_abs = 0.0;
    long over_5pct = 0;
    long missing = 0;
    // per-round breakdown (debug)
    long rcount[5] = {0};   // per-round breakdown (useful for chance subgames)
    double rsum[5] = {0};
};

// `cdeal` is the compound runout index reaching this node (base-ND over deal
// cards), which equals the solver's batch/trainable slot at the node's level.
static void walk(const Subgame& sg, const std::vector<std::vector<float>>& avgs,
                 int nodeid, const json& gnode, int cdeal, int root_round, Stats& st) {
    const Node& nd = sg.nodes[nodeid];
    int ND = sg.ndeals();

    if (nd.type == NT_CHANCE) {
        if (!gnode.contains("dealcards")) return;
        const json& gdeals = gnode["dealcards"];
        int child = nd.children[0];
        for (int dd = 0; dd < ND; ++dd) {
            auto it = gdeals.find(sg.deal_strs[dd]);
            if (it == gdeals.end()) continue;          // card not a valid runout here
            walk(sg, avgs, child, *it, cdeal * ND + dd, root_round, st);
        }
        return;
    }
    if (nd.type != NT_ACTION) return;
    if (!gnode.contains("strategy")) return;

    int slot = cdeal;
    const json& gstrat = gnode["strategy"]["strategy"];   // {handlabel: [probs]}
    int nact = (int)nd.children.size();
    int nc = sg.ncombos(nd.player);
    const std::vector<float>& avg = avgs[nodeid];
    size_t base = (size_t)slot * nact * nc;

    for (int h = 0; h < nc; ++h) {
        const std::string& label = sg.ranges[nd.player][h].label;
        auto it = gstrat.find(label);
        if (it == gstrat.end()) { st.missing++; continue; }
        const json& gp = *it;
        for (int a = 0; a < nact; ++a) {
            double mine = avg[base + (size_t)a * nc + h];
            double theirs = gp[a].get<double>();
            double d = std::fabs(mine - theirs);
            st.count++;
            st.sum_abs += d;
            if (d > st.max_abs) st.max_abs = d;
            if (d > 0.05) st.over_5pct++;
            if (nd.round >= 0 && nd.round < 5) { st.rcount[nd.round]++; st.rsum[nd.round] += d; }
        }
    }

    const json& gkids = gnode["childrens"];
    for (int a = 0; a < nact; ++a) {
        const std::string& label = nd.labels[a];
        int childid = nd.children[a];
        NodeType ct = sg.nodes[childid].type;
        if (ct != NT_ACTION && ct != NT_CHANCE) continue;
        auto it = gkids.find(label);
        if (it == gkids.end()) continue;
        walk(sg, avgs, childid, *it, cdeal, root_round, st);
    }
}

bool compare_to_golden(const Subgame& sg,
                       const std::vector<std::vector<float>>& avgs,
                       const std::string& golden_path,
                       float mean_tol) {
    std::ifstream in(golden_path);
    if (!in) { printf("cannot open golden: %s\n", golden_path.c_str()); return false; }
    json g; in >> g;

    Stats st;
    int root_round = sg.nodes[sg.root].round;
    walk(sg, avgs, sg.root, g, 0, root_round, st);

    double mean = st.count ? st.sum_abs / st.count : 1.0;
    printf("golden comparison: %ld probs compared, mean|diff|=%.5f max|diff|=%.5f "
           ">5%%: %ld, missing-hands: %ld\n",
           st.count, mean, st.max_abs, st.over_5pct, st.missing);
    for (int r = 0; r < 5; ++r)
        if (st.rcount[r]) printf("    round %d: %ld probs, mean|diff|=%.5f\n", r, st.rcount[r], st.rsum[r] / st.rcount[r]);
    return st.count > 0 && mean <= mean_tol;
}

} // namespace texgpu
