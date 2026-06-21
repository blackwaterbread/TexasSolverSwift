#pragma once
#include <string>
#include <vector>
#include "subgame.h"

namespace texgpu {

// Writes the GPU average strategy to a json file for GUI consumption. Per action
// node: acting player, round, pot, action labels, and the per-hand average
// strategy. For chance subgames only the first runout slot (slot 0) is dumped.
void dump_strategy_json(const Subgame& sg,
                        const std::vector<std::vector<float>>& avgs,
                        const std::string& path);

} // namespace texgpu
