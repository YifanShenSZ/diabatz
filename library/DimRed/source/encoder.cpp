#include <DimRed/encoder.hpp>

namespace DimRed {

Encoder::Encoder() {}
Encoder::Encoder(const std::vector<size_t> & dimensions, const bool & symmetric) {
    for (size_t i = 0; i < dimensions.size() - 1; i++) {
        torch::nn::Linear layer = register_module("fc" + std::to_string(i),
            torch::nn::Linear(torch::nn::LinearOptions(dimensions[i], dimensions[i + 1])
                .bias(symmetric)
            ));
        layer->to(torch::kFloat64);
        fcs->push_back(layer);
    }
}
Encoder::~Encoder() {}

void Encoder::freeze(const size_t & NLayers) {
    for (size_t i = 0; i < std::min(NLayers, fcs->size()); i++) {
        auto layer = fcs[i]->as<torch::nn::Linear>();
        layer->weight.set_requires_grad(false);
        layer->  bias.set_requires_grad(false);
    }
}

// Copy the network parameters from `source`
void Encoder::copy_(const std::shared_ptr<Encoder> & source) {
    // The deeper layers are truncated
    for (size_t i = 0; i < std::min(fcs->size(), source->fcs->size()); i++) {
        torch::NoGradGuard no_grad;
        auto layer = fcs[i]->as<torch::nn::Linear>(),
             source_layer = source->fcs[i]->as<torch::nn::Linear>();
        // Check size consistency
        if (layer->weight.size(0) == source_layer->weight.size(0) && layer->weight.size(1) == source_layer->weight.size(1))
        layer->weight.copy_(source_layer->weight);
        if (layer->options.bias() && source_layer->options.bias())
        layer->bias.copy_(source_layer->bias);
    }
}

at::Tensor Encoder::forward(const at::Tensor & x) {
    if (x.sizes().size() != 1) throw std::invalid_argument(
    "DimRed::Encoder::forward: x must be a vector");
    at::Tensor y = x;
    for (size_t i = 0; i < fcs->size() - 1; i++) {
        y = fcs[i]->as<torch::nn::Linear>()->forward(y);
        y = torch::tanh(y);
    }
    y = fcs[fcs->size() - 1]->as<torch::nn::Linear>()->forward(y);
    return y;
}
at::Tensor Encoder::operator()(const at::Tensor & x) {return this->forward(x);}

} // namespace DimRed