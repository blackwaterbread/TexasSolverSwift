#pragma once
#include <string>
#include <vector>
#include "subgame.h"

namespace texgpu {

// Writes the GPU average strategy to a json file for GUI consumption. Per action
// node: acting player, round, pot, action labels, and the per-hand average
// strategy. Nodes with no chance above them carry a flat "strategy" map; nodes
// below one or more chance levels carry a "deals" map keyed by the comma-joined
// runout card labels (e.g. "2c" for turn, "2c,3d" for flop) so every per-runout
// trainset is dumped, not just slot 0.
void dump_strategy_json(const Subgame& sg,
                        const std::vector<std::vector<float>>& avgs,
                        const std::string& path);

} // namespace texgpu
