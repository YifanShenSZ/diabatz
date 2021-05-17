#ifndef train_hpp
#define train_hpp

#include "data.hpp"

namespace trust_region {

void initialize(
const std::shared_ptr<abinitio::DataSet<RegHam>> & regset,
const std::shared_ptr<abinitio::DataSet<DegHam>> & degset);

void optimize(const bool & regularized, const size_t & max_iteration);

} // namespace trust_region

namespace torch_optim {

void Adam();

void SGD();

} // namespace torch_optim

#endif