#ifndef obnet_scalar_hpp
#define obnet_scalar_hpp

#include <torch/torch.h>

namespace obnet {

// The module to hold the neural network for a scalar
struct scalar : torch::nn::Module {
    torch::nn::ModuleList fcs;

    scalar();
    // This copy constructor performs a somewhat deepcopy,
    // where new modules are generated and have same values as `source`
    // We do not use const reference because torch::nn::ModuleList::operator[] does not support `const`,
    // although this constructor would not change `source` of course
    scalar(scalar * source);
    scalar(const std::vector<size_t> & dimensions, const bool & symmetric);
    ~scalar();

    void freeze(const size_t & NLayers = -1);

    at::Tensor forward(const at::Tensor & x);
    at::Tensor operator()(const at::Tensor & x);

    // output hidden layer values before activation to `os`
    void diagnostic(const at::Tensor & x, std::ostream & os);
};

} // namespace obnet

#endif