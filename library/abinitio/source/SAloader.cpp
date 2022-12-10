#include <abinitio/SAloader.hpp>

namespace abinitio {

SAGeomLoader::SAGeomLoader() {}
SAGeomLoader::SAGeomLoader(const int64_t & dimension) : GeomLoader(dimension) {}
SAGeomLoader::~SAGeomLoader() {}

void SAGeomLoader::reset(const int64_t & dimension) {
    GeomLoader::reset(dimension);
    CNPI2point.clear();
    point_defs.clear();
}





SAEnergyLoader::SAEnergyLoader() {}
SAEnergyLoader::SAEnergyLoader(const int64_t & dimension, const int64_t & NStates)
: SAGeomLoader(dimension) {
    energy = at::empty(NStates, {torch::kFloat64});
}
SAEnergyLoader::~SAEnergyLoader() {}

void SAEnergyLoader::reset(const int64_t & dimension, const int64_t & NStates)  {
    SAGeomLoader::reset(dimension);
    energy = at::empty(NStates, {torch::kFloat64});
}





SAHamLoader::SAHamLoader() {}
SAHamLoader::SAHamLoader(const int64_t & dimension, const int64_t & NStates)
: SAEnergyLoader(dimension, NStates) {
    dH = at::empty({NStates, NStates, dimension}, {torch::kFloat64});
}
SAHamLoader::~SAHamLoader() {}

void SAHamLoader::reset(const int64_t & dimension, const int64_t & NStates)  {
    SAEnergyLoader::reset(dimension, NStates);
    dH = at::empty({NStates, NStates, dimension}, {torch::kFloat64});
}

} // namespace abinitio