#include <omp.h>

#include "../../include/global.hpp"

#include "common.hpp"

namespace trust_region {

void set_count() {
    NEqs = 0;
    for (const auto & data : regset) {
        size_t NStates_data = data->NStates();
        // energy least square equations
        NEqs += NStates_data;
        // (▽H)a least square equations
        CL::utility::matrix<at::Tensor> SAdH = data->SAdH();
        for (size_t i = 0; i < NStates_data; i++)
        for (size_t j = i; j < NStates_data; j++)
        NEqs += SAdH[i][j].size(0);
    }
    for (const auto & data : degset) {
        if (NStates != data->NStates()) throw std::invalid_argument(
        "Degenerate data must share a same number of electronic states with "
        "the model to define a comparable composite representation");
        // Hc least square equations
        CL::utility::matrix<size_t> irreds = data->irreds();
        for (size_t i = 0; i < NStates; i++)
        for (size_t j = i; j < NStates; j++)
        if (irreds[i][j] == 0) NEqs++;
        // (▽H)c least square equations
        CL::utility::matrix<at::Tensor> SAdH = data->SAdH();
        for (size_t i = 0; i < NStates; i++)
        for (size_t j = i; j < NStates; j++)
        NEqs += SAdH[i][j].size(0);
    }
    std::cout << "The data set corresponds to " << NEqs << " least square equations\n";
    NPars = 0;
    for (const auto & p : Hdnet->elements->parameters()) NPars += p.numel();
    std::cout << "There are " << NPars << " parameters to train\n\n";
}

void set_parallelism() {
    OMP_NUM_THREADS = omp_get_max_threads();
    std::cout << "The number of threads = " << OMP_NUM_THREADS << '\n';

    Hdnets.resize(OMP_NUM_THREADS);
    Hdnets[0] = Hdnet;
    for (size_t i = 1; i < OMP_NUM_THREADS; i++) {
        Hdnets[i] = std::make_shared<obnet::symat>(Hdnet);
        Hdnets[i]->train();
    }

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
            CL::utility::matrix<at::Tensor> SAdH = data->SAdH();
            for (size_t i = 0; i < NStates_data; i++)
            for (size_t j = i; j < NStates_data; j++)
            segstart[thread] += SAdH[i][j].size(0);
        }
        for (const auto & data : degchunk[thread - 1]) {
            // Hc least square equations
            CL::utility::matrix<size_t> irreds = data->irreds();
            for (size_t i = 0; i < NStates; i++)
            for (size_t j = i; j < NStates; j++)
            if (irreds[i][j] == 0) segstart[thread]++;
            // (▽H)c least square equations
            CL::utility::matrix<at::Tensor> SAdH = data->SAdH();
            for (size_t i = 0; i < NStates; i++)
            for (size_t j = i; j < NStates; j++)
            segstart[thread] += SAdH[i][j].size(0);
        }
    }
    for (size_t thread = 0; thread < OMP_NUM_THREADS; thread++)
    std::cout << "Thread " << thread << " starts with Jacobian row " << segstart[thread] << '\n';
}

void initialize(
const std::shared_ptr<abinitio::DataSet<RegHam>> & _regset,
const std::shared_ptr<abinitio::DataSet<DegHam>> & _degset) {
    NStates = Hdnet->NStates();

    phasers.resize(NStates + 1);
    for (size_t i = 0; i < phasers.size(); i++)
    phasers[i] = std::make_shared<tchem::chem::Phaser>(i);

    regset = _regset->examples();
    degset = _degset->examples();

    set_count();
    set_parallelism();
}

} // namespace trust_region