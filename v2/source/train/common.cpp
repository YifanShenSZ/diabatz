#include "common.hpp"

namespace train {

int64_t NStates;

std::vector<std::shared_ptr<tchem::chem::Phaser>> phasers;

size_t OMP_NUM_THREADS;

// Each thread owns a copy of Hd network
// Thread 0 shares the original Hdnet
std::vector<std::shared_ptr<obnet::symat>> Hdnets;

} // namespace train