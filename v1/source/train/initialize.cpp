#include <omp.h>

#include "common.hpp"

namespace train {

void initialize() {
    NStates = Hdnet->NStates();

    phasers.resize(NStates + 1);
    for (size_t i = 0; i < phasers.size(); i++)
    phasers[i] = std::make_shared<tchem::chem::Phaser>(i);

    OMP_NUM_THREADS = omp_get_max_threads();
    std::cout << "The number of threads = " << OMP_NUM_THREADS << "\n\n";

    Hdnets.resize(OMP_NUM_THREADS);
    Hdnets[0] = Hdnet;
    for (size_t i = 1; i < OMP_NUM_THREADS; i++) {
        Hdnets[i] = std::make_shared<obnet::symat>(Hdnet);
        Hdnets[i]->train();
    }
}

} // namespace train