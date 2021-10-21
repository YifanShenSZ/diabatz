#ifndef CNPI_hpp
#define CNPI_hpp

#include <tchem/intcoord.hpp>

extern std::shared_ptr<tchem::IC::SASICSet> sasicset;

// Given Cartesian coordinate r,
// return CNPI group symmetry adapted internal coordinates and corresponding Jacobians
std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> cart2CNPI(const at::Tensor & r);

// concatenate CNPI group symmetry adapted tensors to point group symmetry adapted tensors
std::vector<at::Tensor> cat(const std::vector<at::Tensor> & xs, const std::vector<std::vector<size_t>> & point2CNPI);

#endif