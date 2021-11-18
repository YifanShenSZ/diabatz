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

void optimize(const bool & regularized, const size_t & max_iteration);

} // namespace trust_region

namespace torch_optim {

void Adam(
const std::shared_ptr<abinitio::DataSet<RegHam>> & regset,
const std::shared_ptr<abinitio::DataSet<DegHam>> & degset,
const std::shared_ptr<abinitio::DataSet<Energy>> & energy_set,
const size_t & max_iteration, const size_t & batch_size, const double & learning_rate,
const std::string & opt_chk
);

void SGD(
const std::shared_ptr<abinitio::DataSet<RegHam>> & regset,
const std::shared_ptr<abinitio::DataSet<DegHam>> & degset,
const std::shared_ptr<abinitio::DataSet<Energy>> & energy_set,
const size_t & max_iteration, const size_t & batch_size, const double & learning_rate,
const std::string & opt_chk
);

} // namespace torch_optim

} // namespace train

#endif