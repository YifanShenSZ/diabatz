#ifndef DimRed_encoder_hpp
#define DimRed_encoder_hpp

#include <torch/torch.h>

namespace DimRed {

struct Encoder : torch::nn::Module {
    torch::nn::ModuleList fcs;

    Encoder();
    // This copy constructor performs a somewhat deepcopy,
    // where new modules are generated and have same values as `source`
    // We do not use const reference because torch::nn::ModuleList::operator[] does not support `const`,
    // although this constructor would not change `source` of course
    Encoder(Encoder * source);
    Encoder(const std::vector<size_t> & dimensions, const bool & symmetric);
    ~Encoder();

    void freeze(const size_t & NLayers = -1);

    // Copy the network parameters from `source`
    void copy_(const std::shared_ptr<Encoder> & source);

    at::Tensor forward(const at::Tensor & x);
    at::Tensor operator()(const at::Tensor & x);
};

} // namespace DimRed

#endif