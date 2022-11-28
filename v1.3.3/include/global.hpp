#ifndef global_hpp
#define global_hpp

#include <SASDIC/SASDICSet.hpp>
#include <obnet/symat.hpp>

#include "InputGenerator.hpp"

extern std::shared_ptr<SASDIC::SASDICSet> sasicset;

// given Cartesian coordinate r,
// return CNPI group symmetry adapted internal coordinates and corresponding Jacobians
std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> cart2CNPI(const at::Tensor & r);

extern std::shared_ptr<obnet::symat> Hdnet1, Hdnet2;

extern std::shared_ptr<InputGenerator> input_generator1, input_generator2;

std::tuple<CL::utility::matrix<at::Tensor>, CL::utility::matrix<at::Tensor>> int2input1(const std::vector<at::Tensor> & qs);
std::tuple<CL::utility::matrix<at::Tensor>, CL::utility::matrix<at::Tensor>> int2input2(const std::vector<at::Tensor> & qs);

extern at::Tensor regularization, prior;

// the "unit" of energy, accounting for the unit difference between energy and gradient
extern double unit, unit_square;

#endif