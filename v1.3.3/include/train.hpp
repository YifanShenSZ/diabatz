#ifndef train_hpp
#define train_hpp

#include "data.hpp"

namespace train {

void initialize();

namespace trust_region {

void initialize(
const std::shared_ptr<abinitio::DataSet<RegHam>> & regset,
const std::shared_ptr<abinitio::DataSet<DegHam>> & degset,
const std::shared_ptr<abinitio::DataSet<Energy>> & energy_set);

void optimize(const size_t & max_iteration);

} // namespace trust_region

} // namespace train

#endif