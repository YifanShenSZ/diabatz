#ifndef global_hpp
#define global_hpp

#include <SASIC/SASICSet.hpp>

extern double threshold;

extern std::shared_ptr<SASIC::SASICSet> sasicset;

// Given Cartesian coordinate r,
// return symmetry adapted internal coordinates
// and their transposed Jacobian over r
std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> cart2int(const at::Tensor & r);

#endif