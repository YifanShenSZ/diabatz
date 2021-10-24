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
    size_t regcount = 0, degcount = 0;
    // the leading threads may have 1 more data point
    size_t leading_regthreads   = regset.size() % OMP_NUM_THREADS,
           leading_regchunksize = regchunksize + 1;
    for (size_t thread = 0; thread < leading_regthreads; thread++) {
        regchunk[thread].resize(leading_regchunksize);
        for (size_t i = 0; i < leading_regchunksize; i++) {
            regchunk[thread][i] = regset[regcount];
            regcount++;
        }
    }
    size_t leading_degthreads   = degset.size() % OMP_NUM_THREADS,
           leading_degchunksize = degchunksize + 1;
    for (size_t thread = 0; thread < leading_degthreads; thread++) {
        degchunk[thread].resize(leading_degchunksize);
        for (size_t i = 0; i < leading_degchunksize; i++) {
            degchunk[thread][i] = degset[degcount];
            degcount++;
        }
    }
    // the remaining threads each has `chunk size` data points
    for (size_t thread = leading_regthreads; thread < OMP_NUM_THREADS; thread++) {
        regchunk[thread].resize(regchunksize);
        for (size_t i = 0; i < regchunksize; i++) {
            regchunk[thread][i] = regset[regcount];
            regcount++;
        }
    }
    for (size_t thread = leading_degthreads; thread < OMP_NUM_THREADS; thread++) {
        degchunk[thread].resize(degchunksize);
        for (size_t i = 0; i < degchunksize; i++) {
            degchunk[thread][i] = degset[degcount];
            degcount++;
        }
    }

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

    for (size_t thread = 0; thread < OMP_NUM_THREADS; thread++) {
        std::cout << "Thread " << thread + 1 << ":\n"
                  << "* owns " << regchunk[thread].size() << " adiabatic data points\n"
                  << "* owns " << degchunk[thread].size() << " composite data points\n"
                  << "* starts with Jacobian row " << segstart[thread] << '\n';
    }
    std::cout << '\n';
}

} // namespace trust_region
} // namespace train