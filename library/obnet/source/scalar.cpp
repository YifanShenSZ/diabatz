#include <obnet/scalar.hpp>

namespace obnet {

scalar::scalar() {}
scalar::scalar(const std::vector<size_t> & dimensions, const bool & symmetric) {
    assert(("The last dimension must be 1 to be a scalar", dimensions.back() == 1));
    for (size_t i = 0; i < dimensions.size() - 1; i++) {
        torch::nn::Linear layer = register_module(
                      "fc" + std::to_string(i), torch::nn::Linear
                      (torch::nn::LinearOptions(dimensions[i], dimensions[i + 1])
                       .bias(symmetric)
                      ));
        fcs->push_back(layer);
    }
}
scalar::~scalar() {}

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