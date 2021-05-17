#include <Foptim/trust_region.hpp>

#include <CppLibrary/linalg.hpp>

#include "../../include/global.hpp"

#include "common.hpp"

namespace trust_region {

void residue (double *  r, const double * c, const int32_t & M, const int32_t & N);
void Jacobian(double * JT, const double * c, const int32_t & M, const int32_t & N);

void regularized_residue (double *  r, const double * c, const int32_t & M, const int32_t & N);
void regularized_Jacobian(double * JT, const double * c, const int32_t & M, const int32_t & N);

void optimize(const bool & regularized, const size_t & max_iteration) {
    double * c = new double[NPars];
    p2c(0, c);
    // Display initial residue
    double * r = new double[NEqs];
    residue(r, c, NEqs, NPars);
    std::cout << "\nThe initial residue = " << CL::linalg::norm2(r, NEqs) << std::endl;
    delete [] r;
    // Run optimization
    if (regularized)
    Foptim::trust_region(regularized_residue, regularized_Jacobian,
                       c, NEqs + NPars, NPars,
                       max_iteration);
    else
    Foptim::trust_region(residue, Jacobian,
                       c, NEqs, NPars,
                       max_iteration);
    c2p(c, 0);
    // Display finial residue
    r = new double[NEqs];
    residue(r, c, NEqs, NPars);
    std::cout << "The final residue = " << CL::linalg::norm2(r, NEqs) << '\n';
    delete [] r;
    // Output
    torch::save(Hdnet->elements, "Hd.net");
    delete [] c;
}

} // namespace trust_region