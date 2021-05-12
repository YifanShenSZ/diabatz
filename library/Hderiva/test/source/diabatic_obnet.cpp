#include <Hderiva/diabatic.hpp>

#include "../include/global.hpp"
#include "../include/data.hpp"

void diabatic_obnet() {
    std::cout << "Testing Hd computed from *obnet*...\n\n";

    Hdnet = std::make_shared<obnet::symat>("obnet_Hd.in");
    Hdnet->to(torch::kFloat64);

    std::vector<std::string> sapoly_files = {"obnet_11.in", "obnet_12.in", "obnet_22.in"};
    input_generator = std::make_shared<InputGenerator>(Hdnet->NStates(), Hdnet->irreds(), sapoly_files, sasicset->NSASICs());

    std::vector<std::string> data = {"min-C1/"};
    std::shared_ptr<abinitio::DataSet<RegHam>> regset;
    std::shared_ptr<abinitio::DataSet<DegHam>> degset;
    std::tie(regset, degset) = read_data(data);

    std::shared_ptr<RegHam> example = regset->get(0);
    std::vector<at::Tensor> qs = example->qs();
    CL::utility::matrix<at::Tensor> xs = example->xs(),
                                 JxqTs = example->JxqTs();
    for (size_t i = 0; i < Hdnet->NStates(); i++)
    for (size_t j = i; j < Hdnet->NStates(); j++)
    xs[i][j].set_requires_grad(true);

    // Assuming that none of off-diagonals is totally symmetric
    CL::utility::matrix<std::vector<at::Tensor>> cs_temp = Hdnet->parameters();
    CL::utility::matrix<at::Tensor> cs(Hdnet->NStates());
    std::vector<at::Tensor> biases(Hdnet->NStates());
    for (size_t i = 0; i < Hdnet->NStates(); i++) {
        biases[i] = cs_temp[i][i][1];
        for (size_t j = i; j < Hdnet->NStates(); j++)
        cs[i][j] = cs_temp[i][j][0].view(cs_temp[i][j][0].numel());
    }

    at::Tensor Hd = Hdnet->forward(xs);

    at::Tensor dxHd = Hderiva::DxHd(Hd, xs, JxqTs, true);
    at::Tensor dxHd_A = dxHd.new_empty(dxHd.sizes());
    for (size_t i = 0; i < Hdnet->NStates(); i++)
    for (size_t j = i; j < Hdnet->NStates(); j++)
    dxHd_A[i][j] = JxqTs[i][j].mv(cs[i][j]);
    double difference = 0.0;
    for (size_t i = 0; i < Hdnet->NStates(); i++)
    for (size_t j = i; j < Hdnet->NStates(); j++)
    difference += (dxHd[i][j] - dxHd_A[i][j]).pow(2).sum().item<double>();
    std::cout << "\nd / dx * Hd: " << sqrt(difference) << '\n';

    at::Tensor dcHd = Hderiva::DcHd(Hd, Hdnet->elements->parameters());
    at::Tensor dcHd_A = dcHd.new_zeros(dcHd.sizes());
    size_t start = 0;
    for (size_t i = 0; i < Hdnet->NStates(); i++) {
        size_t stop = start + xs[i][i].size(0);
        dcHd_A[i][i].slice(0, start, stop) = xs[i][i];
        dcHd_A[i][i][stop] = 1.0;
        start = stop + 1;
        for (size_t j = i + 1; j < Hdnet->NStates(); j++) {
            size_t stop = start + xs[i][j].size(0);
            dcHd_A[i][j].slice(0, start, stop) = xs[i][j];
            start = stop;
        }
    }
    difference = 0.0;
    for (size_t i = 0; i < Hdnet->NStates(); i++)
    for (size_t j = i; j < Hdnet->NStates(); j++)
    difference += (dcHd[i][j] - dcHd_A[i][j]).pow(2).sum().item<double>();
    std::cout << "\nd / dc * Hd: " << sqrt(difference) << '\n';

    at::Tensor dcdxHd = Hderiva::DcDxHd(dxHd, Hdnet->elements->parameters());
    at::Tensor dcdxHd_A = dcdxHd.new_zeros(dcdxHd.sizes());
    start = 0;
    for (size_t i = 0; i < Hdnet->NStates(); i++) {
        size_t stop = start + cs[i][i].size(0);
        dcdxHd_A[i][i].slice(1, start, stop) = JxqTs[i][i];
        dcdxHd_A[i][i].slice(1, stop, stop + 1) = 0.0;
        start = stop + 1;
        for (size_t j = i + 1; j < Hdnet->NStates(); j++) {
            size_t stop = start + cs[i][j].size(0);
            dcdxHd_A[i][j].slice(1, start, stop) = JxqTs[i][j];
            start = stop;
        }
    }
    difference = 0.0;
    for (size_t i = 0; i < Hdnet->NStates(); i++)
    for (size_t j = i; j < Hdnet->NStates(); j++)
    difference += (dcdxHd[i][j] - dcdxHd_A[i][j]).pow(2).sum().item<double>();
    std::cout << "\nd / dc * d / dx * Hd: " << sqrt(difference) << '\n';
}