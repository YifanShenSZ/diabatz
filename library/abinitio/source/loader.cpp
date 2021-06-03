#include <abinitio/loader.hpp>

namespace abinitio {

GeomLoader::GeomLoader() {}
GeomLoader::GeomLoader(const int64_t & dimension) {
    weight = 1.0;
    geom = at::empty(dimension, at::TensorOptions().dtype(torch::kFloat64));
}
GeomLoader::~GeomLoader() {}

void GeomLoader::reset(const int64_t & dimension) {
    weight = 1.0;
    geom = at::empty(dimension, at::TensorOptions().dtype(torch::kFloat64));
}





HamLoader::HamLoader() {}
HamLoader::HamLoader(const int64_t & dimension, const int64_t & NStates)
: GeomLoader(dimension) {
    c10::TensorOptions top = at::TensorOptions().dtype(torch::kFloat64);
    energy = at::empty(NStates, top);
    dH     = at::empty({NStates, NStates, dimension}, top);
}
HamLoader::~HamLoader() {}

void HamLoader::reset(const int64_t & dimension, const int64_t & NStates)  {
    GeomLoader::reset(dimension);
    c10::TensorOptions top = at::TensorOptions().dtype(torch::kFloat64);
    energy = at::empty(NStates, top);
    dH     = at::empty({NStates, NStates, dimension}, top);
}

} // namespace abinitio