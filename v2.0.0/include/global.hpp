#ifndef global_hpp
#define global_hpp

#include <tchem/intcoord.hpp>

#include <DimRed/encoder.hpp>
#include <obnet/symat.hpp>

#include "InputGenerator.hpp"

extern std::shared_ptr<tchem::IC::SASICSet> sasicset;

// Given Cartesian coordinate r,
// return CNPI group symmetry adapted internal coordinates and corresponding Jacobians
std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> cart2int(const at::Tensor & r);

extern std::vector<std::shared_ptr<DimRed::Encoder>> rednets;
extern std::shared_ptr<obnet::symat> Hdnet;

extern std::shared_ptr<InputGenerator> input_generator;

std::tuple<CL::utility::matrix<at::Tensor>, CL::utility::matrix<at::Tensor>> int2input(const std::vector<at::Tensor> & qs);

// the "unit" of energy, accounting for the unit difference between energy and gradient
extern double unit, unit_square;

#endif