#include "common.hpp"

namespace train {

int64_t NStates;

std::shared_ptr<tchem::chem::Orderer> orderer;

size_t OMP_NUM_THREADS;

// each thread owns a copy of Hd network
// thread 0 shares the original Hdnet
std::vector<std::shared_ptr<obnet::symat>> Hdnets;

void initialize() {
    NStates = Hdnet->NStates();

    orderer = std::make_shared<tchem::chem::Orderer>(NStates);

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