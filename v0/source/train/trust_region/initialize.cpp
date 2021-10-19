#include "common.hpp"

namespace train { namespace trust_region {

void initialize(
const std::shared_ptr<abinitio::DataSet<RegHam>> & _regset,
const std::shared_ptr<abinitio::DataSet<DegHam>> & _degset) {
    regset = _regset->examples();
    degset = _degset->examples();

    regchunk.resize(OMP_NUM_THREADS);
    degchunk.resize(OMP_NUM_THREADS);
    size_t regchunksize = regset.size() / OMP_NUM_THREADS,
           degchunksize = degset.size() / OMP_NUM_THREADS;
    std::cout << "Each thread owns " << regchunksize << " data points in adiabatic representation\n"
              << "                 " << degchunksize << " data points in composite representation\n";
    size_t regcount = 0, degcount = 0;
    // Thread 0 to OMP_NUM_THREADS - 2 each owns `chunksize` data points
    for (size_t thread = 0; thread < OMP_NUM_THREADS - 1; thread++) {
        regchunk[thread].resize(regchunksize);
        for (size_t i = 0; i < regchunksize; i++) {
            regchunk[thread][i] = regset[regcount];
            regcount++;
        }
        degchunk[thread].resize(degchunksize);
        for (size_t i = 0; i < degchunksize; i++) {
            degchunk[thread][i] = degset[degcount];
            degcount++;
        }
    }
    // The last thread owns all remaining data points
    regchunk.back().resize(regset.size() - (OMP_NUM_THREADS - 1) * regchunksize);
    for (size_t i = 0; i < regchunk.back().size(); i++) {
        regchunk.back()[i] = regset[regcount];
        regcount++;
    }
    degchunk.back().resize(degset.size() - (OMP_NUM_THREADS - 1) * degchunksize);
    for (size_t i = 0; i < degchunk.back().size(); i++) {
        degchunk.back()[i] = degset[degcount];
        degcount++;
    }
    std::cout << "The last thread owns " << regchunk.back().size() << " data points in adiabatic representation\n"
              << "                     " << degchunk.back().size() << " data points in composite representation\n";

    segstart.resize(OMP_NUM_THREADS);
    segstart[0] = 0;
    for (size_t thread = 1; thread < OMP_NUM_THREADS; thread++) {
        segstart[thread] = segstart[thread - 1];
        for (const auto & data : regchunk[thread - 1]) {
            size_t NStates_data = data->NStates();
            // energy least square equations
            segstart[thread] += NStates_data;
            // (▽H)a least square equations
            for (size_t i = 0; i < NStates_data; i++)
            for (size_t j = i; j < NStates_data; j++)
            segstart[thread] += data->SAdH(i, j).size(0);
        }
        for (const auto & data : degchunk[thread - 1]) {
            // Hc least square equations
            for (size_t i = 0; i < NStates; i++)
            for (size_t j = i; j < NStates; j++)
            if (data->irreds(i, j) == 0) segstart[thread]++;
            // (▽H)c least square equations
            for (size_t i = 0; i < NStates; i++)
            for (size_t j = i; j < NStates; j++)
            segstart[thread] += data->SAdH(i, j).size(0);
        }
    }
    for (size_t thread = 0; thread < OMP_NUM_THREADS; thread++)
    std::cout << "Thread " << thread << " starts with Jacobian row " << segstart[thread] << '\n';
    std::cout << '\n';
}

} // namespace trust_region
} // namespace train