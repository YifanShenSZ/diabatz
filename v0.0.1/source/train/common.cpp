#include "common.hpp"

namespace train {

int64_t NStates;

std::vector<std::shared_ptr<tchem::chem::Phaser>> phasers;

size_t OMP_NUM_THREADS;

// each thread owns a copy of Hd network
// thread 0 shares the original Hdnet
std::vector<std::shared_ptr<obnet::symat>> Hdnets;

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

#include <abinitio/DataSet.hpp>

namespace train { namespace line_search {

// data set
std::vector<std::shared_ptr<RegHam>> regset;
std::vector<std::shared_ptr<DegHam>> degset;
std::vector<std::shared_ptr<Energy>> energy_set;

// each thread owns a chunk of data
std::vector<std::vector<std::shared_ptr<RegHam>>> regchunk;
std::vector<std::vector<std::shared_ptr<DegHam>>> degchunk;
std::vector<std::vector<std::shared_ptr<Energy>>> energy_chunk;

// the container for each thread to accumulate loss and gradient
std::vector<double> losses;
std::vector<at::Tensor> gradients;

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

    for (size_t thread = 0; thread < OMP_NUM_THREADS; thread++) {
        std::cout << "Thread " << thread + 1 << ":\n"
                  << "* owns " << regchunk[thread].size() << " adiabatic data points\n"
                  << "* owns " << degchunk[thread].size() << " composite data points\n"
                  << "* owns " << energy_chunk[thread].size() << " energy-only data points\n";
    }
    std::cout << '\n';

    losses.resize(OMP_NUM_THREADS);
    // cannot allocate memory here: we need to know how many parameters there are
    gradients.resize(OMP_NUM_THREADS);
}

} // namespace line_search
} // namespace train