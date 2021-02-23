#include <obnet/scalar.hpp>

namespace obnet {

scalar::scalar() {}
// This copy constructor performs a somewhat deepcopy,
// where new modules are generated and have same values as `source`
// We do not use const reference because
// torch::nn::ModuleList::operator[] does not support `const`,
// although this constructor would not change `source` of course
scalar::scalar(scalar * source) {
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
scalar::scalar(const std::vector<size_t> & dimensions, const bool & symmetric) {
    assert(("The last dimension must be 1 to be a scalar", dimensions.back() == 1));
    for (size_t i = 0; i < dimensions.size() - 1; i++) {
        torch::nn::Linear layer = register_module("fc" + std::to_string(i),
            torch::nn::Linear(torch::nn::LinearOptions(dimensions[i], dimensions[i + 1])
                .bias(symmetric)
            ));
        layer->to(torch::kFloat64);
        fcs->push_back(layer);
    }
}
scalar::~scalar() {}

void scalar::freeze(const size_t & NLayers) {
    for (size_t i = 0; i < std::min(NLayers, fcs->size()); i++) {
        auto layer = fcs[i]->as<torch::nn::Linear>();
        layer->weight.set_requires_grad(false);
        layer->  bias.set_requires_grad(false);
    }
}

at::Tensor scalar::forward(const at::Tensor & x) {
    assert(("x must be a vector", x.sizes().size() == 1));
    assert(("The dimension of x must match the number of input features",
            x.size(0) == fcs[0]->as<torch::nn::Linear>()->options.in_features()));
    at::Tensor y = x;
    for (auto & layer : (*fcs.get()))
    y = layer->as<torch::nn::Linear>()->forward(y);
    return y[0];
}
at::Tensor scalar::operator()(const at::Tensor & x) {return forward(x);}

} // namespace obnet