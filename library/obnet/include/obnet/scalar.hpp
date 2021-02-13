#ifndef obnet_scalar_hpp
#define obnet_scalar_hpp

#include <torch/torch.h>

namespace obnet {

struct scalar : torch::nn::Module {
    torch::nn::ModuleList fcs;

    scalar();
    scalar(const std::vector<size_t> & dimensions, const bool & symmetric);
    ~scalar();

    at::Tensor forward(const at::Tensor & x);
    at::Tensor operator()(const at::Tensor & x);
};

} // namespace obnet

#endif