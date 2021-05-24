#include <experimental/filesystem>

#include "../../../include/data.hpp"

#include "../common.hpp"
#include "common.hpp"

namespace train { namespace torch_optim {

void SGD(
const std::shared_ptr<abinitio::DataSet<RegHam>> & regset,
const std::shared_ptr<abinitio::DataSet<DegHam>> & degset,
const size_t & max_iteration, const size_t & batch_size, const double & learning_rate,
const std::string & opt_chk) {
    auto reg_loader = torch::data::make_data_loader(* regset,
        torch::data::DataLoaderOptions(batch_size));
    auto deg_loader = torch::data::make_data_loader(* degset,
        torch::data::DataLoaderOptions(batch_size));

    int64_t NPars = 0;
    for (const auto & p : Hdnet->elements->parameters()) NPars += p.numel();
    std::cout << "There are " << NPars << " parameters to train\n\n";
    at::Tensor c = at::empty(NPars, c10::TensorOptions().dtype(torch::kFloat64));
    p2c(0, c.data_ptr<double>());
    // Display initial residue
    std::cout << "The initial residue = "
              << at::cat({reg_residue(regset->examples()), deg_residue(degset->examples())})
                 .norm().item<double>() << '\n' << std::endl;

    torch::optim::SGD optimizer({c}, torch::optim::SGDOptions(learning_rate).momentum(0.9).nesterov(true));
    if (std::experimental::filesystem::exists(opt_chk)) torch::load(optimizer, opt_chk);

    // Create c.grad()
    c.set_requires_grad(true);
    at::Tensor fake_loss = c.dot(c);
    fake_loss.backward();
    c.set_requires_grad(false);

    for (size_t iepoch = 1; iepoch <= max_iteration; iepoch++) {
        for (const auto & batch : * reg_loader) {
            c.grad().copy_(reg_gradient(batch));
            optimizer.step();
            #pragma omp parallel for
            for (size_t thread = 0; thread < OMP_NUM_THREADS; thread++) c2p(c.data_ptr<double>(), thread);
        }
        for (const auto & batch : * deg_loader) {
            c.grad().copy_(deg_gradient(batch));
            optimizer.step();
            #pragma omp parallel for
            for (size_t thread = 0; thread < OMP_NUM_THREADS; thread++) c2p(c.data_ptr<double>(), thread);
        }
        CL::utility::show_time(std::cout);
        std::cout << "epoch " << iepoch << " | residue = "
                  << at::cat({reg_residue(regset->examples()), deg_residue(degset->examples())})
                     .norm().item<double>() << '\n' << std::endl;
        torch::save(Hdnet->elements, std::to_string(iepoch)+"-Hd.net");
        torch::save(optimizer, std::to_string(iepoch)+"-opt.chk");
    }
}

} // namespace torch_optim
} // namespace train