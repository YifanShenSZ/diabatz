#ifndef utility_hpp
#define utility_hpp

#include "global.hpp"

// rescale Hdnet parameters according to feature scaling
// so that with scaled features Hdnet still outputs a same value
void rescale_Hdnet(const CL::utility::matrix<at::Tensor> & avg, const CL::utility::matrix<at::Tensor> & std);

// undo Hdnet parameters scaling
// so that with unscaled features Hdnet still outputs a same value
void unscale_Hdnet(const CL::utility::matrix<at::Tensor> & avg, const CL::utility::matrix<at::Tensor> & std);

#endif