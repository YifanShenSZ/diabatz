#include "global.hpp"

#include "train.hpp"

namespace train {

size_t NStates;

// data set
std::vector<std::shared_ptr<RegHam>> regset;
std::vector<std::shared_ptr<DegHam>> degset;

// the "unit" of energy, accounting for the unit difference between energy and gradient
double unit, unit_square;

std::vector<std::shared_ptr<tchem::chem::Phaser>> phasers;

int32_t OMP_NUM_THREADS;
// Each thread owns a copy of Hd network
// Thread 0 shares the original Hdnet
std::vector<std::shared_ptr<obnet::symat>> Hdnets;
// Each thread owns a chunk of data, works on a segment of residue or Jacobian
// Thread i owns regular data [regchunk[i] - regchunk[0], regchunk[i]),
//            degenerate data [degchunk[i] - degchunk[0], degchunk[i]),
// works on rows [segstart[i], segstart[i + 1]) 
std::vector<size_t> regchunk, degchunk, segstart;

void init_train() {
    NStates = Hdnet->NStates();

    phasers.resize(NStates + 1);
    for (size_t i = 0; i < phasers.size(); i++)
    phasers[i] = std::make_shared<tchem::chem::Phaser>(i);
}

} // namespace train