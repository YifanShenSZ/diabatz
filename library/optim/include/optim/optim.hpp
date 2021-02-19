#ifndef optim_optim_hpp
#define optim_optim_hpp

#include <cstdint>

namespace optim {

extern "C" {
void trust_region_(
    // Required argument
    void (*residue)(double *, const double *, const int &, const int &),
    void (*Jacobian)(double *, const double *, const int &, const int &),
    double * x, const int & M, const int & N,
    // Optional argument
    const int32_t & Warning,
    const int & MaxIteration, const int & MaxStepIteration,
    const double & Precision, const double & MinStepLength
);
}

inline void trust_region(
    void (*residue)(double *, const double *, const int &, const int &),
    void (*Jacobian)(double *, const double *, const int &, const int &),
    double * x, const int & M, const int & N,
    const bool & Warning=true,
    const int & MaxIteration = 1000, const int & MaxStepIteration = 100,
    const double & Precision = 1e-15, const double & MinStepLength = 1e-15
) {
    int32_t w;
    if (Warning) w = -1; else w = 0;
    trust_region_(
        residue, Jacobian, x, M, N,
        w, MaxIteration, MaxStepIteration, Precision, MinStepLength
    );
}

} // namespace optim

#endif