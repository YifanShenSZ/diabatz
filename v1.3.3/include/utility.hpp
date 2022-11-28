#ifndef utility_hpp
#define utility_hpp

#include "global.hpp"

// rescale Hdnet parameters according to feature scaling
// so that with scaled features Hdnet still outputs a same value
void rescale_Hdnet(std::shared_ptr<obnet::symat>& Hdnet, const CL::utility::matrix<at::Tensor> & avg, const CL::utility::matrix<at::Tensor> & std);

// undo Hdnet parameters scaling
// so that with unscaled features Hdnet still outputs a same value
void unscale_Hdnet(std::shared_ptr<obnet::symat>& Hdnet, const CL::utility::matrix<at::Tensor> & avg, const CL::utility::matrix<at::Tensor> & std);

// read parameters from prefix_state1-state2_layer.txt into x
void read_parameters(const std::shared_ptr<obnet::symat>& Hdnet, const std::string & prefix, at::Tensor & x);

// rescale x according to feature scaling
void rescale_parameters(const std::shared_ptr<obnet::symat>& Hdnet, const CL::utility::matrix<at::Tensor> & avg, const CL::utility::matrix<at::Tensor> & std, at::Tensor & x);

#endif