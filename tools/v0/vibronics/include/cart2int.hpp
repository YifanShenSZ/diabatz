#ifndef cart2int_hpp
#define cart2int_hpp

#include <tchem/intcoord.hpp>

extern std::shared_ptr<tchem::IC::SASICSet> sasicset;

// Given Cartesian coordinate r,
// return CNPI group symmetry adapted internal coordinates and corresponding Jacobians
std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> cart2int(const at::Tensor & r);

at::Tensor dHd_cart2int(const at::Tensor & r, const at::Tensor & cartdHd);

#endif