#ifndef abinitio_loader_hpp
#define abinitio_loader_hpp

#include <torch/torch.h>

namespace abinitio {

// Store geometry
struct GeomLoader {
    double weight = 1.0;
    at::Tensor geom;

    GeomLoader();
    GeomLoader(const int64_t & dimension);
    ~GeomLoader();

    void reset(const int64_t & dimension);
};

// Store geometry, adiabatic Hamiltonian and gradient
struct HamLoader : GeomLoader {
    at::Tensor energy, dH;

    HamLoader();
    HamLoader(const int64_t & dimension, const int64_t & NStates);
    ~HamLoader();

    void reset(const int64_t & dimension, const int64_t & NStates);
};

} // namespace abinitio

#endif