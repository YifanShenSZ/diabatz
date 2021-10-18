#include <DimRed/decoder.hpp>

namespace DimRed {

Decoder::Decoder() {}
// This copy constructor performs a somewhat deepcopy,
// where new modules are generated and have same values as `source`
// We do not use const reference because torch::nn::ModuleList::operator[] does not support `const`,
// although this constructor would not change `source` of course
Decoder::Decoder(Decoder * source) {
    torch::NoGradGuard no_grad;
    for (size_t i = 0; i < source->fcs->size(); i++) {
        auto source_layer = source->fcs[i]->as<torch::nn::Linear>();
        torch::nn::Linear layer = register_module("fc" + std::to_string(i),
            torch::nn::Linear(source_layer->options));
        layer->to(torch::kFloat64);
        layer->weight.copy_(source_layer->weight);
        if (layer->options.bias()) layer->bias.copy_(source_layer->bias);
        this->fcs->push_back(layer);
    }
}
Decoder::Decoder(const std::vector<size_t> & dimensions, const bool & symmetric) {
    for (size_t i = 0; i < dimensions.size() - 1; i++) {
        torch::nn::Linear layer = register_module("fc" + std::to_string(i),
            torch::nn::Linear(torch::nn::LinearOptions(dimensions[i], dimensions[i + 1])
                .bias(symmetric)
            ));
        layer->to(torch::kFloat64);
        fcs->push_back(layer);
    }
}
Decoder::~Decoder() {}

void Decoder::freeze(const size_t & NLayers) {
    for (size_t i = 0; i < std::min(NLayers, fcs->size()); i++) {
        auto layer = fcs[i]->as<torch::nn::Linear>();
        layer->weight.set_requires_grad(false);
        layer->  bias.set_requires_grad(false);
    }
}

at::Tensor Decoder::forward(const at::Tensor & x) {
    if (x.sizes().size() != 1) throw std::invalid_argument(
    "DimRed::Decoder::forward: x must be a vector");
    // Q: Why activate input layer?
    // A: As a whole autoencoder, this layer is a hidden, so activation is necessary
    //    The input to decoder is the output from encoder, who does not activate its output
    at::Tensor y = torch::tanh(x);
    for (size_t i = 0; i < fcs->size() - 1; i++) {
        y = fcs[i]->as<torch::nn::Linear>()->forward(y);
        y = torch::tanh(y);
    }
    y = fcs[fcs->size() - 1]->as<torch::nn::Linear>()->forward(y);
    return y;
}
at::Tensor Decoder::operator()(const at::Tensor & x) {return this->forward(x);}

} // namespace DimRed