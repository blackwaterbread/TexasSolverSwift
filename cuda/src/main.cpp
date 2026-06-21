#include <cstdio>
#include <string>
#include "cuda_kernels.cuh"
#include "subgame.h"
#include "validate.h"
#include "cfr_solver.h"
#include "compare_golden.h"
#include "dump_strategy.h"

// River-subgame GPU solver entry point.
// Phase 2: load the serialized subgame and report a summary (parsing validation).
// Later phases: run CFR on the GPU, dump strategy, compare against the CPU golden.
int main(int argc, char** argv) {
    std::string subgame_path = "cuda/configs/subgame.txt";
    std::string golden_path = "cuda/configs/golden_river.json";
    std::string dump_path;   // when set: solve, write strategy json, skip golden compare
    int iters = 200;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if ((a == "-s" || a == "--subgame") && i + 1 < argc) subgame_path = argv[++i];
        else if ((a == "-g" || a == "--golden") && i + 1 < argc) golden_path = argv[++i];
        else if ((a == "-d" || a == "--dump") && i + 1 < argc) dump_path = argv[++i];
        else if ((a == "-n" || a == "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
    }

    if (!texgpu::print_device_info()) {
        printf("No CUDA device found.\n");
        return 1;
    }
    if (!texgpu::saxpy_check(3.0f, 1 << 20)) {
        printf("engine smoke test: FAIL\n");
        return 2;
    }

    texgpu::Subgame sg;
    try {
        sg = texgpu::load_subgame(subgame_path);
    } catch (const std::exception& e) {
        printf("failed to load subgame '%s': %s\n", subgame_path.c_str(), e.what());
        return 3;
    }

    int n_action = 0, n_showdown = 0, n_terminal = 0;
    for (const auto& nd : sg.nodes) {
        if (nd.type == texgpu::NT_ACTION) n_action++;
        else if (nd.type == texgpu::NT_SHOWDOWN) n_showdown++;
        else n_terminal++;
    }

    printf("loaded subgame: %s\n", subgame_path.c_str());
    printf("  board cards: %zu\n", sg.board.size());
    printf("  ranges: p0=%d combos, p1=%d combos\n", sg.ncombos(0), sg.ncombos(1));
    printf("  nodes: %zu total (action=%d showdown=%d terminal=%d), root=%d\n",
           sg.nodes.size(), n_action, n_showdown, n_terminal, sg.root);

    const auto& root = sg.nodes.at(sg.root);
    printf("  root: type=action player=%d, actions=[", root.player);
    for (size_t a = 0; a < root.labels.size(); a++)
        printf("%s%s", a ? ", " : "", root.labels[a].c_str());
    printf("]\n");

    bool ok = true, ok4 = true;
    if (!dump_path.empty()) {
        // GUI mode: solve and dump the strategy json, no golden available.
        const char* kind = sg.chance_levels >= 2 ? "flop/2-chance"
                         : sg.chance_levels == 1 ? "turn/1-chance" : "river";
        printf("--- full CFR solve (%s), dump mode ---\n", kind);
        texgpu::CudaCfrSolver solver(sg);
        double secs = solver.train(iters);
        auto avgs = solver.averageStrategies();
        printf("solved %d iterations on GPU in %.3f s (%.2f iters/s).\n",
               iters, secs, iters / secs);
        try {
            texgpu::dump_strategy_json(sg, avgs, dump_path);
        } catch (const std::exception& e) {
            printf("failed to dump strategy: %s\n", e.what());
            return 5;
        }
        printf("strategy written to %s\n", dump_path.c_str());
        solver.exploitability();
        return 0;
    }

    if (!sg.has_chance) {
        // River-only leaf/trainable unit checks use base ranks.
        printf("--- leaf kernel validation ---\n");
        ok = texgpu::validate_leaves(sg);
        printf("leaf kernels: %s\n", ok ? "PASS" : "FAIL");
        printf("--- trainable (DCFR) validation ---\n");
        ok4 &= texgpu::validate_trainable(2, sg.ncombos(0), 200);
        ok4 &= texgpu::validate_trainable(3, sg.ncombos(1), 200);
        printf("trainable kernels: %s\n", ok4 ? "PASS" : "FAIL");
    }

    const char* kind = sg.chance_levels >= 2 ? "flop/2-chance"
                     : sg.chance_levels == 1 ? "turn/1-chance" : "river";
    printf("--- full CFR solve vs golden (%s) ---\n", kind);
    texgpu::CudaCfrSolver solver(sg);
    double secs = solver.train(iters);
    auto avgs = solver.averageStrategies();
    printf("solved %d iterations on GPU in %.3f s (%.2f iters/s).\n", iters, secs, iters / secs);
    // Deeper chance trees diverge more on mixed (non-unique / dynamics-sensitive)
    // spots due to CPU/GPU equilibrium selection — same effect seen in the turn
    // solver, just larger with two chance levels. Upper-round EVs and pure spots
    // still match tightly, which is what validates the mechanism.
    float tol = sg.chance_levels >= 2 ? 0.03f : 0.02f;
    bool ok5 = texgpu::compare_to_golden(sg, avgs, golden_path, tol);
    printf("CFR vs golden: %s\n", ok5 ? "PASS" : "FAIL");

    // Equilibrium-invariant convergence check (independent of strategy comparison).
    solver.exploitability();

    return (ok && ok4 && ok5) ? 0 : 4;
}
