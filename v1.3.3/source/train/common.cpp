#include "common.hpp"

namespace train {

int64_t NStates;

std::shared_ptr<tchem::chem::Orderer> orderer;

size_t OMP_NUM_THREADS;

// each thread owns a copy of Hd network
// thread 0 shares the original Hdnet
std::vector<std::shared_ptr<obnet::symat>> Hdnet1s, Hdnet2s;

void initialize() {
    NStates = Hdnet1->NStates();

    orderer = std::make_shared<tchem::chem::Orderer>(NStates);

    OMP_NUM_THREADS = omp_get_max_threads();
    std::cout << "The number of threads = " << OMP_NUM_THREADS << "\n\n";

    Hdnet1s.resize(OMP_NUM_THREADS);
    Hdnet1s[0] = Hdnet1;
    Hdnet2s.resize(OMP_NUM_THREADS);
    Hdnet2s[0] = Hdnet2;
    for (size_t i = 1; i < OMP_NUM_THREADS; i++) {
        Hdnet1s[i] = std::make_shared<obnet::symat>(Hdnet1);
        Hdnet1s[i]->train();
        Hdnet2s[i] = std::make_shared<obnet::symat>(Hdnet2);
        Hdnet2s[i]->train();
    }
}

} // namespace train

#include <abinitio/DataSet.hpp>

namespace train { namespace trust_region {

// data set
std::vector<std::shared_ptr<RegHam>> regset;
std::vector<std::shared_ptr<DegHam>> degset;
std::vector<std::shared_ptr<Energy>> energy_set;

// Each thread owns a chunk of data
std::vector<std::vector<std::shared_ptr<RegHam>>> regchunk;
std::vector<std::vector<std::shared_ptr<DegHam>>> degchunk;
std::vector<std::vector<std::shared_ptr<Energy>>> energy_chunk;

// Each thread works on a segment of residue or Jacobian
// Thread i works on rows [segstart[i], segstart[i + 1])
std::vector<size_t> segstart;

void initialize(
const std::shared_ptr<abinitio::DataSet<RegHam>> & _regset,
const std::shared_ptr<abinitio::DataSet<DegHam>> & _degset,
const std::shared_ptr<abinitio::DataSet<Energy>> & _energy_set) {
    regset = _regset->examples();
    degset = _degset->examples();
    energy_set = _energy_set->examples();

    regchunk.resize(OMP_NUM_THREADS);
    degchunk.resize(OMP_NUM_THREADS);
    energy_chunk.resize(OMP_NUM_THREADS);
    size_t regchunksize = regset.size() / OMP_NUM_THREADS,
           degchunksize = degset.size() / OMP_NUM_THREADS,
           energy_chunksize = energy_set.size() / OMP_NUM_THREADS;
    size_t regcount = 0, degcount = 0, energy_count = 0;
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
    size_t leading_energy_threads   = energy_set.size() % OMP_NUM_THREADS,
           leading_energy_chunksize = energy_chunksize + 1;
    for (size_t thread = 0; thread < leading_energy_threads; thread++) {
        energy_chunk[thread].resize(leading_energy_chunksize);
        for (size_t i = 0; i < leading_energy_chunksize; i++) {
            energy_chunk[thread][i] = energy_set[energy_count];
            energy_count++;
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
    for (size_t thread = leading_energy_threads; thread < OMP_NUM_THREADS; thread++) {
        energy_chunk[thread].resize(energy_chunksize);
        for (size_t i = 0; i < energy_chunksize; i++) {
            energy_chunk[thread][i] = energy_set[energy_count];
            energy_count++;
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
        for (const auto & data : energy_chunk[thread - 1]) {
            // energy least square equations
            segstart[thread] += data->NStates();
        }
    }

    for (size_t thread = 0; thread < OMP_NUM_THREADS; thread++) {
        std::cout << "Thread " << thread + 1 << ":\n"
                  << "* owns " << regchunk[thread].size() << " adiabatic data points\n"
                  << "* owns " << degchunk[thread].size() << " composite data points\n"
                  << "* owns " << energy_chunk[thread].size() << " energy-only data points\n"
                  << "* starts with Jacobian row " << segstart[thread] << '\n';
    }
    std::cout << '\n';
}

} // namespace trust_region
} // namespace train