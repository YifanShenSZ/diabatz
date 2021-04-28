#include <DimRed/encoder.hpp>
#include <DimRed/decoder.hpp>

int main() {
    std::vector<size_t> endims = {8, 4, 2}, dedims = {2, 4, 8};
    auto encoder = std::make_shared<DimRed::Encoder>(endims, true);
    auto decoder = std::make_shared<DimRed::Decoder>(dedims, true);

    auto parameters = encoder->parameters(),
         depars = decoder->parameters();
    parameters.insert(parameters.end(), depars.begin(), depars.end());

    torch::optim::Adam optimizer(parameters);

    for (size_t iepoch = 1; iepoch <= 10; iepoch++) {
        double rmsd = 0.0;
        for (size_t irepeat = 0; irepeat < 10; irepeat++) {
            at::Tensor x = at::rand(8, c10::TensorOptions().dtype(torch::kFloat64));
            at::Tensor loss = torch::mse_loss(
                                  decoder->forward(encoder->forward(x)),
                                  x,
                                  at::Reduction::Sum
                              );
            rmsd += loss.item<double>();
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
        }
        std::cout << "epoch = " << iepoch
                  << ", loss = " << sqrt(rmsd / 9.0)
                  << '\n';
    }
}