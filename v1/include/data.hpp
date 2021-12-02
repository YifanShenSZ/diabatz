#ifndef data_hpp
#define data_hpp

#include <abinitio/DataSet.hpp>

#include "data_classes.hpp"

std::tuple<std::shared_ptr<abinitio::DataSet<RegHam>>, std::shared_ptr<abinitio::DataSet<DegHam>>>
read_data(const std::vector<std::string> & user_list, const double & zero_point);

std::shared_ptr<abinitio::DataSet<Energy>> read_energy(const std::vector<std::string> & user_list, const double & zero_point);

#endif