#include <CppLibrary/utility.hpp>

#include "../include/global.hpp"

void SGD(const size_t & max_iteration, const size_t & batch_size, const double & learning_rate) {
    std::cout << "Adopt SGD optimizer\n";

    auto parameters = encoder->parameters(),
         depars     = decoder->parameters();
    parameters.insert(parameters.end(), depars.begin(), depars.end());
    torch::optim::SGD optimizer(parameters, torch::optim::SGDOptions(learning_rate).momentum(0.9).nesterov(true));
    
    auto geom_loader = torch::data::make_data_loader(* geom_set,
        torch::data::DataLoaderOptions(batch_size).drop_last(true));
    std::cout << "batch size = " << batch_size << std::endl;
    size_t follow = max_iteration / 10;
    for (size_t iepoch = 1; iepoch <= max_iteration; iepoch++) {
        for (const auto & batch : * geom_loader) {
            at::Tensor loss = at::zeros({}, c10::TensorOptions().dtype(torch::kFloat64));
            for (const auto & data : batch) {
                const auto & x = data->qs()[irreducible];
                loss += torch::mse_loss(decoder->forward(encoder->forward(x)), x, at::Reduction::Sum);
            }
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
        }
        if (iepoch % follow == 0) {
            CL::utility::show_time(std::cout);
            std::cout << "epoch = " << iepoch
                      << ", RMSD = " << RMSD() << '\n' << std::endl;
        }
    }
}