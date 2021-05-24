#include "../../../include/data.hpp"

#include "../common.hpp"
#include "common.hpp"

namespace train { namespace torch_optim {

void SGD(
const std::shared_ptr<abinitio::DataSet<RegHam>> & regset,
const std::shared_ptr<abinitio::DataSet<DegHam>> & degset,
const size_t & max_iteration, const size_t & batch_size, const double & learning_rate
) {
    auto reg_loader = torch::data::make_data_loader(* regset,
        torch::data::DataLoaderOptions(batch_size));
    auto deg_loader = torch::data::make_data_loader(* degset,
        torch::data::DataLoaderOptions(batch_size));

    int64_t NPars = 0;
    for (const auto & p : Hdnet->elements->parameters()) NPars += p.numel();
    std::cout << "There are " << NPars << " parameters to train\n\n";
    at::Tensor c = at::rand(NPars, c10::TensorOptions().dtype(torch::kFloat64));
    torch::optim::SGD optimizer({c}, torch::optim::SGDOptions(learning_rate).momentum(0.9).nesterov(true));

    // Create c.grad()
    c.set_requires_grad(true);
    at::Tensor fake_loss = c.dot(c);
    fake_loss.backward();
    c.set_requires_grad(false);

    for (size_t iepoch = 1; iepoch <= max_iteration; iepoch++) {
        for (const auto & batch : * reg_loader) {
            c.grad().copy_(reg_gradient(batch));
            optimizer.step();
        }
        std::cout << "epoch " << iepoch << " | residue = " << reg_residue(regset->examples()).norm().item<double>() << std::endl;
    }
}

} // namespace torch_optim
} // namespace train