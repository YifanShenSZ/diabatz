#ifndef global_hpp
#define global_hpp

#include <tchem/intcoord.hpp>

extern std::shared_ptr<tchem::IC::SASICSet> sasicset;

// Given Cartesian coordinate r,
// return CNPI group symmetry adapted internal coordinates and corresponding Jacobians
std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> cart2CNPI(const at::Tensor & r);

#endif