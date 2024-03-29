#include <obnet/scalar.hpp>

namespace obnet {

scalar::scalar() {}
// This copy constructor performs a somewhat deepcopy,
// where new modules are generated and have same values as `source`
// We do not use const reference because torch::nn::ModuleList::operator[] does not support `const`,
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
    if (dimensions.back() != 1) throw std::invalid_argument(
    "obnet::scalar: The last dimension must be 1 to be a scalar");
    for (size_t i = 0; i < dimensions.size() - 1; i++) {
        torch::nn::Linear layer = register_module("fc" + std::to_string(i),
            torch::nn::Linear(torch::nn::LinearOptions(dimensions[i], dimensions[i + 1])
                .bias(symmetric)
            ));
        layer->to(torch::kFloat64);
        // use Xavier initialization with gain = 5/3 since we adopt tanh activation
        torch::nn::init::xavier_uniform_(layer->weight, 5.0 / 3.0);
        fcs->push_back(layer);
    }
    // Assuming that the input layer has magnitude of ~ 1,
    // with Xavier or Kaiming initialization the output also has magnitude of ~ 1
    // From chemical experience, a quantity usually takes ~ 0.01 in atomic unit,
    // so we additionally shrink the output layer to that range
    {
        torch::NoGradGuard no_grad;
        fcs[fcs->size() - 1]->as<torch::nn::Linear>()->weight *= 0.01;
    }
}
scalar::~scalar() {}

void scalar::freeze(const size_t & NLayers) {
    for (size_t i = 0; i < std::min(NLayers, fcs->size()); i++) {
        auto layer = fcs[i]->as<torch::nn::Linear>();
        layer->weight.requires_grad_(false);
	if (layer->options.bias()) layer->bias.requires_grad_(false);
    }
}

at::Tensor scalar::forward(const at::Tensor & x) {
    if (x.sizes().size() != 1) throw std::invalid_argument(
    "obnet::scalar::forward: x must be a vector");
    at::Tensor y = x;
    for (size_t i = 0; i < fcs->size() - 1; i++) {
        y = fcs[i]->as<torch::nn::Linear>()->forward(y);
        y = torch::tanh(y);
    }
    y = fcs[fcs->size() - 1]->as<torch::nn::Linear>()->forward(y);
    return y[0];
}
at::Tensor scalar::operator()(const at::Tensor & x) {return this->forward(x);}

// output weighed features and hidden layer values before activation to `os`
void scalar::diagnostic(const at::Tensor & x, std::ostream & os) {
    if (x.sizes().size() != 1) throw std::invalid_argument(
    "obnet::scalar::forward: x must be a vector");
    torch::NoGradGuard no_grad;
    at::Tensor y = x;
    for (size_t i = 0; i < fcs->size() - 1; i++) {
        os << "hidden layer " << i + 1 << ":\n";
        at::Tensor weight = fcs[i]->as<torch::nn::Linear>()->weight;
        at::Tensor weighed_features = (weight * y).transpose(0, 1);
        os << "transposed weighed features:\n" << weighed_features << '\n';
        y = fcs[i]->as<torch::nn::Linear>()->forward(y);
        os << "hidden layer before activation:\n" << y << '\n';
        y = torch::tanh(y);
    }
    os << "output layer:\n";
    at::Tensor weight = fcs[fcs->size() - 1]->as<torch::nn::Linear>()->weight;
    at::Tensor weighed_features = (weight * y).transpose(0, 1);
    os << "transposed weighed features:\n" << weighed_features << '\n';
    y = fcs[fcs->size() - 1]->as<torch::nn::Linear>()->forward(y);
    os << "final output:\n" << y << '\n';
}

} // namespace obnet
