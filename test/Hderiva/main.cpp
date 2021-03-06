#include <CppLibrary/argparse.hpp>
#include <CppLibrary/utility.hpp>

#include <Hderiva/diabatic.hpp>

#include "global.hpp"
#include "data.hpp"

int main(const size_t & argc, const char ** & argv) {
    sasicset = std::make_shared<tchem::IC::SASICSet>("default", "IntCoordDef", "SAS.in");

    Hdnet = std::make_shared<obnet::symat>("Hd.in");
    Hdnet->to(torch::kFloat64);

    std::vector<std::string> sapoly_files = {"11.in", "12.in", "22.in"};
    input_generator = std::make_shared<InputGenerator>(Hdnet->NStates(), sapoly_files, sasicset->NSASICs());

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
    std::cout << "\nd / dx * Hd: "
              << (dxHd - dxHd_A).norm().item<double>() << '\n';

    at::Tensor dcHd = Hderiva::DcHd(Hd, Hdnet->elements->parameters());
    for (size_t i = 0    ; i < Hdnet->NStates(); i++)
    for (size_t j = i + 1; j < Hdnet->NStates(); j++)
    dcHd[j][i] = 0.0;
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
    std::cout << "\nd / dc * Hd: "
              << (dcHd - dcHd_A).norm().item<double>() << '\n';

    at::Tensor dcdxHd = Hderiva::DcDxHd(dxHd, Hdnet->elements->parameters());
    for (size_t i = 0    ; i < Hdnet->NStates(); i++)
    for (size_t j = i + 1; j < Hdnet->NStates(); j++)
    dcdxHd[j][i] = 0.0;
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
    std::cout << "\nd / dc * d / dx * Hd: "
              << (dcdxHd - dcdxHd_A).norm().item<double>() << '\n';
}