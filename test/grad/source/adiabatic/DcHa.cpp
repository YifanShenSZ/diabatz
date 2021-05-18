#include <tchem/linalg.hpp>

#include "../Hd.hpp"
#include "../commutor.hpp"

// Return d / dc of Ha matrix elements, adiabatic energies and states
std::tuple<at::Tensor, at::Tensor, at::Tensor,
at::Tensor, at::Tensor> DcHa
(const at::Tensor & c00, const at::Tensor & c01, const at::Tensor & c11, const at::Tensor & q) {
    assert(("c00 must require gradient", c00.requires_grad()));
    assert(("c01 must require gradient", c01.requires_grad()));
    assert(("c11 must require gradient", c11.requires_grad()));
    at::Tensor Hd = libHd::Hd(c00, c01, c11, q);
//Hd[1][0] = Hd[0][1];
    at::Tensor energies, states;
    std::tie(energies, states) = Hd.symeig(true);
// This misbehaviour has been recorded in Torch-Chemistry issue #1
// tchem::linalg::UT_sy_U who explicitly loops over matrix elements
// deteriorates the backward propagation
// where the diagonal gradient works fine,
// but the off-diagonal gradient is erroneous
at::Tensor Ha = tchem::linalg::UT_sy_U(Hd, states);
// Using mm works out fine
//at::Tensor Ha = states.transpose(0, 1).mm(Hd.mm(states));
    at::Tensor dHa_c00 = q.new_zeros({2, 2, 3}),
               dHa_c01 = q.new_zeros({2, 2, 2}),
               dHa_c11 = q.new_zeros({2, 2, 3});
    for (size_t i = 0; i < 2; i++)
    for (size_t j = i; j < 2; j++) {
        torch::autograd::variable_list g = torch::autograd::grad({Ha[i][j]}, {c00, c01, c11}, {}, true);
        dHa_c00[i][j].copy_(g[0]);
        dHa_c01[i][j].copy_(g[1]);
        dHa_c11[i][j].copy_(g[2]);
    }
    return std::make_tuple(dHa_c00, dHa_c01, dHa_c11, energies, states);
}

// Analytical version
std::tuple<at::Tensor, at::Tensor, at::Tensor> analytical_DcHa
(const at::Tensor & c00, const at::Tensor & c01, const at::Tensor & c11, const at::Tensor & q,
const at::Tensor & energies, const at::Tensor & states) {
    // dH / dc in diabatic representation
    at::Tensor dHd_c00, dHd_c01, dHd_c11;
    std::tie(dHd_c00, dHd_c01, dHd_c11) = libHd::analytical_DcHd(c00, c01, c11, q);
    // nac
    at::Tensor nac_c00 = tchem::linalg::UT_sy_U(dHd_c00, states),
               nac_c01 = tchem::linalg::UT_sy_U(dHd_c01, states),
               nac_c11 = tchem::linalg::UT_sy_U(dHd_c11, states);
    nac_c00[0][1] /= energies[1] - energies[0];
    nac_c01[0][1] /= energies[1] - energies[0];
    nac_c11[0][1] /= energies[1] - energies[0];
    // Combine to d / dc * Ha
    at::Tensor dHa_dc00 = tchem::linalg::UT_sy_U(dHd_c00, states)
                        + commutor(energies.diag(), nac_c00),
               dHa_dc01 = tchem::linalg::UT_sy_U(dHd_c01, states)
                        + commutor(energies.diag(), nac_c01),
               dHa_dc11 = tchem::linalg::UT_sy_U(dHd_c11, states)
                        + commutor(energies.diag(), nac_c11);
    return std::make_tuple(dHa_dc00, dHa_dc01, dHa_dc11);
}

void test_DcHa() {
    c10::TensorOptions top = at::TensorOptions().dtype(torch::kFloat64);
    at::Tensor c00 = at::rand(3, top),
               c01 = at::rand(2, top),
               c11 = at::rand(3, top);
    c00.set_requires_grad(true);
    c01.set_requires_grad(true);
    c11.set_requires_grad(true);
    at::Tensor q = at::rand(2, top);
    q.set_requires_grad(true);
    at::Tensor dHa_c00, dHa_c01, dHa_c11, energies, states;
    std::tie(dHa_c00, dHa_c01, dHa_c11, energies, states) = DcHa(c00, c01, c11, q);
    at::Tensor dHa_c00_A, dHa_c01_A, dHa_c11_A;
    std::tie(dHa_c00_A, dHa_c01_A, dHa_c11_A) = analytical_DcHa(c00, c01, c11, q, energies, states);
    std::cout << "\nd / dc * Ha: autograd incorrectly gives non-zero off-diag, see Torch-Chemistry issue #1\n"
              << (dHa_c00 - dHa_c00_A).norm().item<double>() << "    "
              << (dHa_c01 - dHa_c01_A).norm().item<double>() << "    "
              << (dHa_c11 - dHa_c11_A).norm().item<double>() << '\n';
    std::cout << "Erroroneous term: ||d / dc01 * Ha[0][1]|| = "
              << dHa_c01[0][1].norm().item<double>() << '\n';
}