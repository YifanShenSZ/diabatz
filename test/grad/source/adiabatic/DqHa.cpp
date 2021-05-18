#include <tchem/linalg.hpp>

#include "../Hd.hpp"
#include "../commutor.hpp"

// Return d / dq of Ha matrix elements, adiabatic energies and states
std::tuple<at::Tensor, at::Tensor, at::Tensor> DqHa
(const at::Tensor & c00, const at::Tensor & c01, const at::Tensor & c11, const at::Tensor & q) {
    assert(("q must require gradient", q.requires_grad()));
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
    at::Tensor dHa = q.new_zeros({2, 2, 2});
    for (size_t i = 0; i < 2; i++)
    for (size_t j = i; j < 2; j++) {
        torch::autograd::variable_list g = torch::autograd::grad({Ha[i][j]}, {q}, {}, true);
        dHa[i][j].copy_(g[0]);
    }
    return std::make_tuple(dHa, energies, states);
}

// Analytical version
at::Tensor analytical_DqHa(const at::Tensor & c00, const at::Tensor & c01, const at::Tensor & c11, const at::Tensor & q,
const at::Tensor & energies, const at::Tensor & states) {
    // dH / dq in diabatic representation
    at::Tensor dHd = libHd::analytical_DqHd(c00, c01, c11, q);
    // nac
    at::Tensor nac = tchem::linalg::UT_sy_U(dHd, states);
    nac[0][1] /= energies[1] - energies[0];
    // Combine to d / dq * Ha
    at::Tensor dHa = tchem::linalg::UT_sy_U(dHd, states)
                   + commutor(energies.diag(), nac);
    return dHa;
}

void test_DqHa() {
    c10::TensorOptions top = at::TensorOptions().dtype(torch::kFloat64);
    at::Tensor c00 = at::rand(3, top),
               c01 = at::rand(2, top),
               c11 = at::rand(3, top);
    c00.set_requires_grad(true);
    c01.set_requires_grad(true);
    c11.set_requires_grad(true);
    at::Tensor q = at::rand(2, top);
    q.set_requires_grad(true);
    at::Tensor dHa, energies, states;
    std::tie(dHa, energies, states) = DqHa(c00, c01, c11, q);
    at::Tensor dHa_A = analytical_DqHa(c00, c01, c11, q, energies, states);
    std::cout << "\nd / dq * Ha: autograd incorrectly gives non-zero off-diag, see Torch-Chemistry issue #1\n"
              << (dHa - dHa_A).norm().item<double>() << '\n';
    std::cout << "Erroroneous term: ||d / dq * Ha[0][1]|| = "
              << dHa[0][1].norm().item<double>() << '\n';
}