#include <omp.h>

#include <CppLibrary/linalg.hpp>

#include <Fopt/Fopt.hpp>

#include "global.hpp"

#include "train.hpp"

namespace train {

int64_t NStates;

std::vector<std::shared_ptr<tchem::chem::Phaser>> phasers;

// data set
std::vector<std::shared_ptr<RegHam>> regset;
std::vector<std::shared_ptr<DegHam>> degset;

// the "unit" of energy, accounting for the unit difference between energy and gradient
double unit, unit_square;

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

void set_unit() {
    assert(("Data set must have been provided", ! regset.empty()));
    double maxe = 0.0, maxg = 0.0;
    for (const auto & data : regset) {
        double temp = data->energy()[0].item<double>();
        maxe = temp > maxe ? temp : maxe;
        temp = data->dH()[0][0].abs().max().item<double>();
        maxg = temp > maxg ? temp : maxg;
    }
    std::cout << "maximum ground state energy = " << maxe << '\n'
              << "maximum ||ground state energy gradient||_infinity = " << maxg << '\n'; 
    if (maxe > 0.0) unit = maxg / maxe;
    else            unit = 1.0; // fail safe
    std::cout << "so set gradient / energy scaling to " << unit << "\n\n";
    unit_square = unit * unit;
}

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
        assert(("Data must share a same number of electronic states with "
                "the model to define a comparable composite representation",
                NStates = data->NStates()));
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
    // Thread 0 to OMP_NUM_THREADS - 2 each owns `chunksize` data
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
    // The last thread owns all remaining data
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

} // namespace train

void initialize(
const std::shared_ptr<abinitio::DataSet<RegHam>> & regset,
const std::shared_ptr<abinitio::DataSet<DegHam>> & degset) {
    train::NStates = Hdnet->NStates();

    train::phasers.resize(train::NStates + 1);
    // Make phasers for cases from 2 to NStates states
    for (size_t i = 2; i < train::phasers.size(); i++)
    train::phasers[i] = std::make_shared<tchem::chem::Phaser>(i);

    train::regset = regset->examples();
    train::degset = degset->examples();

    train::set_unit();
    train::set_count();
    train::set_parallelism();
}

namespace train {
void residue (double *  r, const double * c, const int32_t & M, const int32_t & N);
void Jacobian(double * JT, const double * c, const int32_t & M, const int32_t & N);

void regularized_residue (double *  r, const double * c, const int32_t & M, const int32_t & N);
void regularized_Jacobian(double * JT, const double * c, const int32_t & M, const int32_t & N);
} // namespace train

void optimize(const bool & regularized, const size_t & max_iteration) {
    double * c = new double[train::NPars];
    train::p2c(0, c);
    // Display initial residue
    double * r = new double[train::NEqs];
    train::residue(r, c, train::NEqs, train::NPars);
    std::cout << "\nThe initial residue = " << CL::linalg::norm2(r, train::NEqs) << std::endl;
    delete [] r;
    // Run optimization
    if (regularized)
    Fopt::trust_region(train::regularized_residue, train::regularized_Jacobian,
                       c, train::NEqs + train::NPars, train::NPars,
                       max_iteration);
    else
    Fopt::trust_region(train::residue, train::Jacobian,
                       c, train::NEqs, train::NPars,
                       max_iteration);
    train::c2p(c, 0);
    // Display finial residue
    r = new double[train::NEqs];
    train::residue(r, c, train::NEqs, train::NPars);
    std::cout << "The final residue = " << CL::linalg::norm2(r, train::NEqs) << '\n';
    delete [] r;
    // Output
    torch::save(Hdnet->elements, "Hd.net");
    delete [] c;
}