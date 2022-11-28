#include <Foptim/Foptim.hpp>

#include <CppLibrary/linalg.hpp>

#include "common.hpp"

namespace train { namespace trust_region {

std::tuple<int32_t, int32_t> count_eq_par() {
    int32_t NEqs = 0;
    for (const auto & data : regset) {
        size_t NStates_data = data->NStates();
        // energy least square equations
        NEqs += NStates_data;
        // (▽H)a least square equations
        for (size_t i = 0; i < NStates_data; i++)
        for (size_t j = i; j < NStates_data; j++)
        NEqs += data->SAdH(i, j).size(0);
    }
    for (const auto & data : degset) {
        if (NStates != data->NStates()) throw std::invalid_argument(
        "Degenerate data must share a same number of electronic states with "
        "the model to define a comparable composite representation");
        // Hc least square equations
        for (size_t i = 0; i < NStates; i++)
        for (size_t j = i; j < NStates; j++)
        if (data->irreds(i, j) == 0) NEqs++;
        // (▽H)c least square equations
        for (size_t i = 0; i < NStates; i++)
        for (size_t j = i; j < NStates; j++)
        NEqs += data->SAdH(i, j).size(0);
    }
    for (const auto & data : energy_set) {
        // energy least square equations
        NEqs += data->NStates();
    }
    std::cout << "The data set corresponds to " << NEqs << " least square equations\n";

    int32_t NPars = 0;
    for (const auto & p : Hdnet1->elements->parameters()) NPars += p.numel();
    for (const auto & p : Hdnet2->elements->parameters()) NPars += p.numel();
    std::cout << "There are " << NPars << " parameters to train\n\n";

    return std::make_tuple(NEqs, NPars);
}

void residue (double *  r, const double * c, const int32_t & M, const int32_t & N);
void Jacobian(double * JT, const double * c, const int32_t & M, const int32_t & N);

void regularized_residue (double *  r, const double * c, const int32_t & M, const int32_t & N);
void regularized_Jacobian(double * JT, const double * c, const int32_t & M, const int32_t & N);

void optimize(const size_t & max_iteration) {
    int32_t NEqs, NPars;
    std::tie(NEqs, NPars) = count_eq_par();

    double * c = new double[NPars];
    p2c(0, c);
    // Display initial residue
    double * r = new double[NEqs];
    residue(r, c, NEqs, NPars);
    std::cout << "The initial residue = " << CL::linalg::norm2(r, NEqs) << std::endl;
    delete [] r;

    Foptim::trust_region_verbose(regularized_residue, regularized_Jacobian,
                                 c, NEqs + NPars, NPars,
                                 max_iteration);
    c2p(c, 0);

    r = new double[NEqs];
    residue(r, c, NEqs, NPars);
    std::cout << "The final residue = " << CL::linalg::norm2(r, NEqs) << '\n';
    delete [] r;
    delete [] c;
}

} // namespace trust_region
} // namespace train