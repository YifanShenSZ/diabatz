#ifndef utility_hpp
#define utility_hpp

#include "global.hpp"

// read parameters from prefix_state1-state2_layer.txt into x
void read_parameters(const std::string & prefix, at::Tensor & x);

#endif