#include "dump_strategy.h"

#include <fstream>
#include <stdexcept>
#include "json.hpp"

using json = nlohmann::json;

namespace texgpu {

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

        json jstrat = json::object();
        for (int h = 0; h < nc; h++) {
            json probs = json::array();
            for (int a = 0; a < nact; a++) {
                // slot 0 layout: (0*nact + a)*nc + h
                size_t idx = (size_t)a * nc + h;
                float v = (idx < av.size()) ? av[idx] : 0.0f;
                probs.push_back(v);
            }
            jstrat[sg.ranges[nd.player][h].label] = probs;
        }
        jn["strategy"] = jstrat;
        jnodes[std::to_string(i)] = jn;
    }
    root["nodes"] = jnodes;

    std::ofstream f(path);
    if (!f) throw std::runtime_error("cannot open dump file: " + path);
    f << root.dump(1);
}

} // namespace texgpu
