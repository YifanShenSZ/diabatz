#ifndef train_hpp
#define train_hpp

#include "data.hpp"

namespace train {

void initialize();

namespace torch_optim {

void SGD(
const std::shared_ptr<abinitio::DataSet<RegHam>> & regset,
const std::shared_ptr<abinitio::DataSet<DegHam>> & degset,
const std::shared_ptr<abinitio::DataSet<Energy>> & energy_set,
const size_t & max_iteration, const size_t & batch_size,
const double & learning_rate, const double & weight_decay,
const std::string & opt_chk
);

void NAG(
const std::shared_ptr<abinitio::DataSet<RegHam>> & regset,
const std::shared_ptr<abinitio::DataSet<DegHam>> & degset,
const std::shared_ptr<abinitio::DataSet<Energy>> & energy_set,
const size_t & max_iteration, const size_t & batch_size,
const double & learning_rate, const double & weight_decay,
const std::string & opt_chk
);

void Adam(
const std::shared_ptr<abinitio::DataSet<RegHam>> & regset,
const std::shared_ptr<abinitio::DataSet<DegHam>> & degset,
const std::shared_ptr<abinitio::DataSet<Energy>> & energy_set,
const size_t & max_iteration, const size_t & batch_size,
const double & learning_rate, const double & weight_decay,
const std::string & opt_chk
);

} // namespace torch_optim

} // namespace train

#endif