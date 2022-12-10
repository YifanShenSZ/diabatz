#include <abinitio/loader.hpp>

namespace abinitio {

GeomLoader::GeomLoader() {}
GeomLoader::GeomLoader(const int64_t & dimension) {
    weight = 1.0;
    geom = at::empty(dimension, {torch::kFloat64});
}
GeomLoader::~GeomLoader() {}

void GeomLoader::reset(const int64_t & dimension) {
    weight = 1.0;
    geom = at::empty(dimension, {torch::kFloat64});
}





EnergyLoader::EnergyLoader() {}
EnergyLoader::EnergyLoader(const int64_t & dimension, const int64_t & NStates)
: GeomLoader(dimension) {
    energy = at::empty(NStates, {torch::kFloat64});
}
EnergyLoader::~EnergyLoader() {}

void EnergyLoader::reset(const int64_t & dimension, const int64_t & NStates)  {
    GeomLoader::reset(dimension);
    energy = at::empty(NStates, {torch::kFloat64});
}





HamLoader::HamLoader() {}
HamLoader::HamLoader(const int64_t & dimension, const int64_t & NStates)
: EnergyLoader(dimension, NStates) {
    dH = at::empty({NStates, NStates, dimension}, {torch::kFloat64});
}
HamLoader::~HamLoader() {}

void HamLoader::reset(const int64_t & dimension, const int64_t & NStates)  {
    EnergyLoader::reset(dimension, NStates);
    dH = at::empty({NStates, NStates, dimension}, {torch::kFloat64});
}

} // namespace abinitio