#pragma once
#include <string>
#include <vector>
#include "subgame.h"

namespace texgpu {

// Compares GPU average strategies (per node id) against the CPU golden JSON,
// matching nodes by action-label path and hands by label. Prints stats.
// Returns true if the mean abs difference is within mean_tol.
bool compare_to_golden(const Subgame& sg,
                       const std::vector<std::vector<float>>& avgs,
                       const std::string& golden_path,
                       float mean_tol = 0.02f);

} // namespace texgpu
