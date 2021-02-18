// global variable

#ifndef global_hpp
#define global_hpp

#include <tchem/SASintcoord.hpp>

#include <obnet/symat.hpp>

#include "Hd.hpp"

extern std::shared_ptr<tchem::IC::SASICSet> sasicset;

// Given Cartesian coordinate r,
// return CNPI group symmetry adapted internal coordinates and corresponding Jacobians
std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> cart2int(const at::Tensor & r);

extern std::shared_ptr<InputGenerator> input_generator;

CL::utility::matrix<at::Tensor> int2input(const std::vector<at::Tensor> & qs);

extern std::shared_ptr<obnet::symat> Hdnet;

#endif