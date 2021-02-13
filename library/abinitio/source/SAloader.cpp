#include <abinitio/SAloader.hpp>

namespace abinitio {

SAGeomLoader::SAGeomLoader() {}
SAGeomLoader::SAGeomLoader(const int64_t & dimension)
: GeomLoader(dimension) {}
SAGeomLoader::~SAGeomLoader() {}

void SAGeomLoader::reset(const int64_t & dimension) {
    GeomLoader::reset(dimension);
    CNPI2point.resize(0);
}





SAHamLoader::SAHamLoader() {}
SAHamLoader::SAHamLoader(const int64_t & dimension, const int64_t & NStates)
: SAGeomLoader(dimension) {
    c10::TensorOptions top = at::TensorOptions().dtype(torch::kFloat64);
    energy = at::empty(NStates, top);
    dH     = at::empty({NStates, NStates, dimension}, top);
}
SAHamLoader::~SAHamLoader() {}

void SAHamLoader::reset(const int64_t & dimension, const int64_t & NStates)  {
    SAGeomLoader::reset(dimension);
    c10::TensorOptions top = at::TensorOptions().dtype(torch::kFloat64);
    energy = at::empty(NStates, top);
    dH     = at::empty({NStates, NStates, dimension}, top);
}

} // namespace abinitio