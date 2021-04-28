#ifndef DimRed_decoder_hpp
#define DimRed_decoder_hpp

#include <torch/torch.h>

namespace DimRed {

struct Decoder : torch::nn::Module {
    torch::nn::ModuleList fcs;

    Decoder();
    Decoder(const std::vector<size_t> & dimensions, const bool & symmetric);
    ~Decoder();

    void freeze(const size_t & NLayers = -1);

    at::Tensor forward(const at::Tensor & x);
    at::Tensor operator()(const at::Tensor & x);
};

} // namespace DimRed

#endif