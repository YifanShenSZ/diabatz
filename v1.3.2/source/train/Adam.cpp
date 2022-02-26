#include <experimental/filesystem>

#include "../../include/data.hpp"

#include "common.hpp"

namespace train { namespace torch_optim {

void Adam(
const std::shared_ptr<abinitio::DataSet<RegHam>> & regset,
const std::shared_ptr<abinitio::DataSet<DegHam>> & degset,
const std::shared_ptr<abinitio::DataSet<Energy>> & energy_set,
const size_t & max_iteration, const size_t & batch_size,
const double & learning_rate, const double & weight_decay,
const std::string & opt_chk) {
    auto reg_loader = torch::data::make_data_loader(* regset,
        torch::data::DataLoaderOptions(batch_size).drop_last(true));
    auto deg_loader = torch::data::make_data_loader(* degset,
        torch::data::DataLoaderOptions(batch_size).drop_last(true));
    auto energy_loader = torch::data::make_data_loader(* energy_set,
        torch::data::DataLoaderOptions(batch_size).drop_last(true));

    int64_t NPars = 0;
    for (const auto & p : Hdnet->elements->parameters()) NPars += p.numel();
    std::cout << "There are " << NPars << " parameters to train\n\n";
    at::Tensor c = at::empty(NPars, c10::TensorOptions().dtype(torch::kFloat64));
    p2c(0, c.data_ptr<double>());
    // display initial residue
    std::cout << "The initial residue = "
              << at::cat({reg_residue(regset->examples()),
                          deg_residue(degset->examples()),
                          energy_residue(energy_set->examples())
                 })
                 .norm().item<double>() << std::endl;

    torch::optim::Adam optimizer({c}, torch::optim::AdamOptions(learning_rate).weight_decay(weight_decay));
    if (std::experimental::filesystem::exists(opt_chk)) torch::load(optimizer, opt_chk);

    // create c.grad()
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
        for (const auto & batch : * energy_loader) {
            c.grad().copy_(energy_gradient(batch));
            optimizer.step();
            #pragma omp parallel for
            for (size_t thread = 0; thread < OMP_NUM_THREADS; thread++) c2p(c.data_ptr<double>(), thread);
        }
        std::cout << '\n';
        CL::utility::show_time(std::cout);
        std::cout << "epoch " << iepoch << " | residue = "
                  << at::cat({reg_residue(regset->examples()),
                              deg_residue(degset->examples()),
                              energy_residue(energy_set->examples())
                     })
                     .norm().item<double>() << std::endl;
    }
    // Hdnet will be saved in main because of feature scaling
    torch::save(optimizer, "opt.chk");
}

} // namespace torch_optim
} // namespace train