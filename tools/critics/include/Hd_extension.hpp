#ifndef Hd_extension_hpp
#define Hd_extension_hpp

#include <torch/torch.h>

at::Tensor compute_ddHd(const at::Tensor & r);

at::Tensor compute_energy(const at::Tensor & r);

std::tuple<at::Tensor, at::Tensor> compute_energy_dHa(const at::Tensor & r);

at::Tensor compute_ddHa(const at::Tensor & r);

#endif