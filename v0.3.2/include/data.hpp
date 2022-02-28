#ifndef data_hpp
#define data_hpp

#include <abinitio/DataSet.hpp>

#include "data_classes.hpp"

std::tuple<std::shared_ptr<abinitio::DataSet<RegHam>>, std::shared_ptr<abinitio::DataSet<DegHam>>>
read_data(const std::vector<std::string> & user_list, const double & zero_point);

std::shared_ptr<abinitio::DataSet<Energy>> read_energy(const std::vector<std::string> & user_list, const double & zero_point);

// given a regular data set
// return a shift and a width for feature scaling
std::tuple<CL::utility::matrix<at::Tensor>, CL::utility::matrix<at::Tensor>>
statisticize_regset(const std::shared_ptr<abinitio::DataSet<RegHam>> & regset);

#endif