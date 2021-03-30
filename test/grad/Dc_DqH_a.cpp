#include <tchem/linalg.hpp>
#include <tchem/chemistry.hpp>

#include "Hd.hpp"
#include "commutor.hpp"

// Return d / dc * (d / dq * H)a matrix elements,
// (d / dq * H)a, adiabatic energies and states
std::tuple<at::Tensor, at::Tensor, at::Tensor,
at::Tensor, at::Tensor, at::Tensor> Dc_DqH_a
(const at::Tensor & c00, const at::Tensor & c01, const at::Tensor & c11, const at::Tensor & q) {
    assert(("c00 must require gradient", c00.requires_grad()));
    assert(("c01 must require gradient", c01.requires_grad()));
    assert(("c11 must require gradient", c11.requires_grad()));
    assert(("q must require gradient", q.requires_grad()));
    // (d / dq * H)a
    at::Tensor Hd = libHd::Hd(c00, c01, c11, q);
    at::Tensor DqHd = q.new_zeros({2, 2, 2});
    for (size_t i = 0; i < 2; i++)
    for (size_t j = i; j < 2; j++) {
        torch::autograd::variable_list g = torch::autograd::grad({Hd[i][j]}, {q}, {}, true, true);
        DqHd[i][j] = g[0];
    }
    at::Tensor energies, states;
    std::tie(energies, states) = Hd.symeig(true);
    at::Tensor DqH_a = tchem::linalg::UT_sy_U(DqHd, states);
    // d / dc * (d / dq * H)a
    at::Tensor Dc00_DqH_a = DqH_a.new_zeros({2, 2, 2, 3}),
               Dc01_DqH_a = DqH_a.new_zeros({2, 2, 2, 2}),
               Dc11_DqH_a = DqH_a.new_zeros({2, 2, 2, 3});
    for (size_t i = 0; i < 2; i++)
    for (size_t j = i; j < 2; j++)
    for (size_t k = 0; k < 2; k++) {
        torch::autograd::variable_list g = torch::autograd::grad({DqH_a[i][j][k]}, {c00, c01, c11}, {}, true);
        Dc00_DqH_a[i][j][k].copy_(g[0]);
        Dc01_DqH_a[i][j][k].copy_(g[1]);
        Dc11_DqH_a[i][j][k].copy_(g[2]);
    }
    return std::make_tuple(Dc00_DqH_a, Dc01_DqH_a, Dc11_DqH_a, DqH_a, energies, states);
}

// Analytical version
std::tuple<at::Tensor, at::Tensor, at::Tensor> analytical_Dc_DqH_a
(const at::Tensor & c00, const at::Tensor & c01, const at::Tensor & c11, const at::Tensor & q,
const at::Tensor & energies, const at::Tensor & states) {
    // d / dc * d / dq * H in diabatic representation
    at::Tensor Dc00DqHd, Dc01DqHd, Dc11DqHd;
    std::tie(Dc00DqHd, Dc01DqHd, Dc11DqHd) = libHd::analytical_DcDqHd(c00, c01, c11, q);
    // (d / dq * H)a
    at::Tensor DqHd = libHd::analytical_DqHd(c00, c01, c11, q);
    at::Tensor DqH_a = tchem::linalg::UT_sy_U(DqHd, states);
    // nac
    at::Tensor Dc00Hd, Dc01Hd, Dc11Hd;
    std::tie(Dc00Hd, Dc01Hd, Dc11Hd) = libHd::analytical_DcHd(c00, c01, c11, q);
    at::Tensor nac_c00 = tchem::linalg::UT_sy_U(Dc00Hd, states),
               nac_c01 = tchem::linalg::UT_sy_U(Dc01Hd, states),
               nac_c11 = tchem::linalg::UT_sy_U(Dc11Hd, states);
    nac_c00[0][1] /= energies[1] - energies[0];
    nac_c01[0][1] /= energies[1] - energies[0];
    nac_c11[0][1] /= energies[1] - energies[0];
    // d / dc * (dH / dq)a
    at::Tensor Dc00_DqH_a = tchem::linalg::UT_sy_U(Dc00DqHd, states)
                          + commutor(DqH_a, nac_c00),
               Dc01_DqH_a = tchem::linalg::UT_sy_U(Dc01DqHd, states)
                          + commutor(DqH_a, nac_c01),
               Dc11_DqH_a = tchem::linalg::UT_sy_U(Dc11DqHd, states)
                          + commutor(DqH_a, nac_c11);
    return std::make_tuple(Dc00_DqH_a, Dc01_DqH_a, Dc11_DqH_a);
}

// Numerical version
std::tuple<at::Tensor, at::Tensor, at::Tensor> numerical_Dc_DqH_a
(const at::Tensor & c00, const at::Tensor & c01, const at::Tensor & c11, const at::Tensor & q,
const at::Tensor & DqH_a) {
    tchem::chem::Phaser phaser(2);
    at::Tensor Dc00_DqH_a = q.new_zeros({2, 2, 2, 3});
    for (size_t l = 0; l < 3; l++) {
        at::Tensor c00_plus = c00.clone();
        c00_plus[l] += 1e-5;
        at::Tensor DqH_a_plus = libHd::DqH_a(c00_plus, c01, c11, q);
        phaser.fix_ob_(DqH_a_plus, DqH_a);
        at::Tensor c00_minus = c00.clone();
        c00_minus[l] -= 1e-5;
        at::Tensor DqH_a_minus = libHd::DqH_a(c00_minus, c01, c11, q);
        phaser.fix_ob_(DqH_a_minus, DqH_a);
        for (size_t i = 0; i < 2; i++)
        for (size_t j = i; j < 2; j++)
        for (size_t k = 0; k < 2; k++)
        Dc00_DqH_a[i][j][k][l] = (DqH_a_plus[i][j][k] - DqH_a_minus[i][j][k]) / 2e-5;
    }
    at::Tensor Dc01_DqH_a = q.new_zeros({2, 2, 2, 2});
    for (size_t l = 0; l < 2; l++) {
        at::Tensor c01_plus = c01.clone();
        c01_plus[l] += 1e-5;
        at::Tensor DqH_a_plus = libHd::DqH_a(c00, c01_plus, c11, q);
        phaser.fix_ob_(DqH_a_plus, DqH_a);
        at::Tensor c01_minus = c01.clone();
        c01_minus[l] -= 1e-5;
        at::Tensor DqH_a_minus = libHd::DqH_a(c00, c01_minus, c11, q);
        phaser.fix_ob_(DqH_a_minus, DqH_a);
        for (size_t i = 0; i < 2; i++)
        for (size_t j = i; j < 2; j++)
        for (size_t k = 0; k < 2; k++)
        Dc01_DqH_a[i][j][k][l] = (DqH_a_plus[i][j][k] - DqH_a_minus[i][j][k]) / 2e-5;
    }
    at::Tensor Dc11_DqH_a = q.new_zeros({2, 2, 2, 3});
    for (size_t l = 0; l < 3; l++) {
        at::Tensor c11_plus = c11.clone();
        c11_plus[l] += 1e-5;
        at::Tensor DqH_a_plus = libHd::DqH_a(c00, c01, c11_plus, q);
        phaser.fix_ob_(DqH_a_plus, DqH_a);
        at::Tensor c11_minus = c11.clone();
        c11_minus[l] -= 1e-5;
        at::Tensor DqH_a_minus = libHd::DqH_a(c00, c01, c11_minus, q);
        phaser.fix_ob_(DqH_a_minus, DqH_a);
        for (size_t i = 0; i < 2; i++)
        for (size_t j = i; j < 2; j++)
        for (size_t k = 0; k < 2; k++)
        Dc11_DqH_a[i][j][k][l] = (DqH_a_plus[i][j][k] - DqH_a_minus[i][j][k]) / 2e-5;
    }
    return std::make_tuple(Dc00_DqH_a, Dc01_DqH_a, Dc11_DqH_a);
}

void test_Dc_DqH_a() {
    c10::TensorOptions top = at::TensorOptions().dtype(torch::kFloat64);
    at::Tensor c00 = at::rand(3, top),
               c01 = at::rand(2, top),
               c11 = at::rand(3, top);
    c00.set_requires_grad(true);
    c01.set_requires_grad(true);
    c11.set_requires_grad(true);
    at::Tensor q = at::rand(2, top);
    q.set_requires_grad(true);
    at::Tensor Dc00_DqH_a, Dc01_DqH_a, Dc11_DqH_a, DqH_a, energies, states;
    std::tie(Dc00_DqH_a, Dc01_DqH_a, Dc11_DqH_a, DqH_a, energies, states) = Dc_DqH_a(c00, c01, c11, q);
    at::Tensor Dc00_DqH_a_A, Dc01_DqH_a_A, Dc11_DqH_a_A;
    std::tie(Dc00_DqH_a_A, Dc01_DqH_a_A, Dc11_DqH_a_A) = analytical_Dc_DqH_a(c00, c01, c11, q, energies, states);
    at::Tensor Dc00_DqH_a_N, Dc01_DqH_a_N, Dc11_DqH_a_N;
    std::tie(Dc00_DqH_a_N, Dc01_DqH_a_N, Dc11_DqH_a_N) = numerical_Dc_DqH_a(c00, c01, c11, q, DqH_a);
    std::cout << "\nd / dc * (d / dq * H)a: probably also arises from Torch-Chemistry issue #1\n"
              << "autograd vs numerical: "
              << (Dc00_DqH_a - Dc00_DqH_a_N).norm().item<double>() << "    "
              << (Dc01_DqH_a - Dc01_DqH_a_N).norm().item<double>() << "    "
              << (Dc11_DqH_a - Dc11_DqH_a_N).norm().item<double>() << '\n'
              << "analytical vs numerical: "
              << (Dc00_DqH_a_A - Dc00_DqH_a_N).norm().item<double>() << "    "
              << (Dc01_DqH_a_A - Dc01_DqH_a_N).norm().item<double>() << "    "
              << (Dc11_DqH_a_A - Dc11_DqH_a_N).norm().item<double>() << '\n';
    std::cout << "Entire d / dc01 * (d / dq * H)a is erroneous, including diags and off-diags\n";
}