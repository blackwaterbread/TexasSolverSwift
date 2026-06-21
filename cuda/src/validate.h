#pragma once
#include "subgame.h"

namespace texgpu {

// Runs GPU-vs-CPU brute-force checks on every showdown/terminal node in the
// subgame, for both players, with a deterministic pseudo-random opponent reach.
// Returns true if all leaf payoffs match within tolerance.
bool validate_leaves(const Subgame& sg, float tol = 1e-3f);

// Runs a DCFR trainable on GPU and on a CPU mirror over an identical sequence of
// pseudo-random regrets, comparing the final average strategy. Returns true on match.
bool validate_trainable(int nact, int ncards, int iters, float tol = 1e-4f);

} // namespace texgpu
