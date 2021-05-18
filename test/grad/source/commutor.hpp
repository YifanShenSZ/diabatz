#ifndef commutor_hpp
#define commutor_hpp

#include <torch/torch.h>

at::Tensor commutor(const at::Tensor & A, const at::Tensor M);

#endif