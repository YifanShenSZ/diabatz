#ifndef global_hpp
#define global_hpp

#include <SASDIC/SASDICSet.hpp>

#include <obnet/symat.hpp>

#include "InputGenerator.hpp"

extern std::shared_ptr<SASDIC::SASDICSet> sasicset;

// Given Cartesian coordinate r,
// return CNPI group symmetry adapted internal coordinates and corresponding Jacobians
std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> cart2int(const at::Tensor & r);

extern std::shared_ptr<obnet::symat> Hdnet;

extern std::shared_ptr<InputGenerator> input_generator;

std::tuple<CL::utility::matrix<at::Tensor>, CL::utility::matrix<at::Tensor>> int2input(const std::vector<at::Tensor> & qs);

#endif