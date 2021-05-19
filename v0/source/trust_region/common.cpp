#include "common.hpp"

namespace trust_region {

int64_t NStates;

std::vector<std::shared_ptr<tchem::chem::Phaser>> phasers;

// data set
std::vector<std::shared_ptr<RegHam>> regset;
std::vector<std::shared_ptr<DegHam>> degset;

// Number of least square equations and fitting parameters
int32_t NEqs, NPars;

size_t OMP_NUM_THREADS;
// Each thread owns a copy of Hd network
// Thread 0 shares the original Hdnet
std::vector<std::shared_ptr<obnet::symat>> Hdnets;
// Each thread owns a chunk of data
std::vector<std::vector<std::shared_ptr<RegHam>>> regchunk;
std::vector<std::vector<std::shared_ptr<DegHam>>> degchunk;
// Each thread works on a segment of residue or Jacobian
// Thread i works on rows [segstart[i], segstart[i + 1])
std::vector<size_t> segstart;

} // namespace trust_region