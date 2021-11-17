#ifndef abinitio_SAloader_hpp
#define abinitio_SAloader_hpp

#include <torch/torch.h>

#include <abinitio/loader.hpp>

namespace abinitio {

// store geometry and symmetry information
struct SAGeomLoader : GeomLoader {
    // mapping from CNPI group to point group
    // CNPI irreducible i becomes point irreducible CNPI2point[i]
    std::vector<size_t> CNPI2point;
    // point group symmetry adapted internal coordinate definitions
    std::vector<std::string> point_defs;

    SAGeomLoader();
    SAGeomLoader(const int64_t & dimension);
    ~SAGeomLoader();

    void reset(const int64_t & dimension);
};

// store geometry and symmetry information, energy
struct SAEnergyLoader : SAGeomLoader {
    at::Tensor energy;

    SAEnergyLoader();
    SAEnergyLoader(const int64_t & dimension, const int64_t & NStates);
    ~SAEnergyLoader();

    void reset(const int64_t & dimension, const int64_t & NStates);
};

// store geometry and symmetry information, adiabatic energy and gradient
struct SAHamLoader : SAEnergyLoader {
    at::Tensor dH;

    SAHamLoader();
    SAHamLoader(const int64_t & dimension, const int64_t & NStates);
    ~SAHamLoader();

    void reset(const int64_t & dimension, const int64_t & NStates);
};

} // namespace abinitio

#endif