#ifndef train_hpp
#define train_hpp

#include "data.hpp"

namespace train {

void initialize();

namespace trust_region {

void initialize(
const std::shared_ptr<abinitio::DataSet<RegHam>> & regset,
const std::shared_ptr<abinitio::DataSet<DegHam>> & degset);

void optimize(const bool & regularized, const size_t & max_iteration);

} // namespace trust_region

namespace torch_optim {

void Adam(
const std::shared_ptr<abinitio::DataSet<RegHam>> & regset,
const std::shared_ptr<abinitio::DataSet<DegHam>> & degset,
const size_t & max_iteration, const size_t & batch_size, const double & learning_rate
);

void SGD(
const std::shared_ptr<abinitio::DataSet<RegHam>> & regset,
const std::shared_ptr<abinitio::DataSet<DegHam>> & degset,
const size_t & max_iteration, const size_t & batch_size, const double & learning_rate
);

} // namespace torch_optim

} // namespace train

#endif