#ifndef train_hpp
#define train_hpp

#include "data.hpp"

namespace train {

void initialize();

namespace line_search {

void initialize(
const std::shared_ptr<abinitio::DataSet<RegHam>> & regset,
const std::shared_ptr<abinitio::DataSet<DegHam>> & degset,
const std::shared_ptr<abinitio::DataSet<Energy>> & energy_set);

void optimize(const bool & regularized, const std::string & optimizer, const double & learning_rate, const int32_t & memory, const size_t & max_iteration);

} // namespace line_search

} // namespace train

#endif