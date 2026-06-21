// Console entry for the CPU solver, used to (1) produce the golden strategy
// reference and (2) later serialize a river subgame for the GPU engine.
// Mirrors the logic of src/console.cpp's main_backup but is a real main().
#include "include/tools/CommandLineTool.h"
#include "include/tools/argparse.hpp"

int main(int argc, const char** argv) {
    ArgumentParser parser;
    parser.addArgument("-i", "--input_file", 1, true);
    parser.addArgument("-r", "--resource_dir", 1, true);
    parser.addArgument("-m", "--mode", 1, true);
    parser.parse(argc, argv);

    string input_file = parser.retrieve<string>("input_file");
    string resource_dir = parser.retrieve<string>("resource_dir");
    if (resource_dir.empty()) resource_dir = "./resources";
    string mode = parser.retrieve<string>("mode");
    if (mode.empty()) mode = "holdem";
    if (mode != "holdem" && mode != "shortdeck")
        throw runtime_error(tfm::format("mode %s error, not in ['holdem','shortdeck']", mode));

    CommandLineTool clt = CommandLineTool(mode, resource_dir);
    if (input_file.empty()) {
        clt.startWorking();
    } else {
        cout << "EXEC FROM FILE: " << input_file << endl;
        clt.execFromFile(input_file);
    }
    return 0;
}
