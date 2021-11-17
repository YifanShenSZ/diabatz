#ifndef abinitio_loader_hpp
#define abinitio_loader_hpp

#include <torch/torch.h>

namespace abinitio {

// store geometry
struct GeomLoader {
    double weight = 1.0;
    at::Tensor geom;

    GeomLoader();
    GeomLoader(const int64_t & dimension);
    ~GeomLoader();

    void reset(const int64_t & dimension);
};

// store geometry, energy
struct EnergyLoader : GeomLoader {
    at::Tensor energy;

    EnergyLoader();
    EnergyLoader(const int64_t & dimension, const int64_t & NStates);
    ~EnergyLoader();

    void reset(const int64_t & dimension, const int64_t & NStates);
};

// store geometry, adiabatic energy and gradient
struct HamLoader : EnergyLoader {
    at::Tensor dH;

    HamLoader();
    HamLoader(const int64_t & dimension, const int64_t & NStates);
    ~HamLoader();

    void reset(const int64_t & dimension, const int64_t & NStates);
};

} // namespace abinitio

#endif