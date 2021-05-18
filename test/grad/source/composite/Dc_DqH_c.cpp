#include <tchem/linalg.hpp>
#include <tchem/chemistry.hpp>

#include "../Hd.hpp"
#include "../commutor.hpp"

// Return d / dc * (d / dq * H)c matrix elements,
// (d / dq * H)c, eigenvalues and eigenvectors
std::tuple<at::Tensor, at::Tensor, at::Tensor,
at::Tensor, at::Tensor, at::Tensor> Dc_DqH_c
(const at::Tensor & c00, const at::Tensor & c01, const at::Tensor & c11, const at::Tensor & q) {
    assert(("c00 must require gradient", c00.requires_grad()));
    assert(("c01 must require gradient", c01.requires_grad()));
    assert(("c11 must require gradient", c11.requires_grad()));
    assert(("q must require gradient", q.requires_grad()));
    // (d / dq * H)c
    at::Tensor Hd = libHd::Hd(c00, c01, c11, q);
    at::Tensor DqHd = q.new_zeros({2, 2, 2});
    for (size_t i = 0; i < 2; i++)
    for (size_t j = i; j < 2; j++) {
        torch::autograd::variable_list g = torch::autograd::grad({Hd[i][j]}, {q}, {}, true, true);
        DqHd[i][j] = g[0];
    }
    at::Tensor DqHdDqHd = tchem::linalg::sy3matdotmul(DqHd, DqHd);
    at::Tensor eigvals, eigvecs;
    std::tie(eigvals, eigvecs) = DqHdDqHd.symeig(true);
    at::Tensor DqH_c = tchem::linalg::UT_sy_U(DqHd, eigvecs);
    // d / dc * (d / dq * H)c
    at::Tensor Dc00_DqH_c = DqH_c.new_zeros({2, 2, 2, 3}),
               Dc01_DqH_c = DqH_c.new_zeros({2, 2, 2, 2}),
               Dc11_DqH_c = DqH_c.new_zeros({2, 2, 2, 3});
    for (size_t i = 0; i < 2; i++)
    for (size_t j = i; j < 2; j++)
    for (size_t k = 0; k < 2; k++) {
        torch::autograd::variable_list g = torch::autograd::grad({DqH_c[i][j][k]}, {c00, c01, c11}, {}, true);
        Dc00_DqH_c[i][j][k].copy_(g[0]);
        Dc01_DqH_c[i][j][k].copy_(g[1]);
        Dc11_DqH_c[i][j][k].copy_(g[2]);
    }
    return std::make_tuple(Dc00_DqH_c, Dc01_DqH_c, Dc11_DqH_c, DqH_c, eigvals, eigvecs);
}

// Analytical version
std::tuple<at::Tensor, at::Tensor, at::Tensor> analytical_Dc_DqH_c
(const at::Tensor & c00, const at::Tensor & c01, const at::Tensor & c11, const at::Tensor & q,
const at::Tensor & eigvals, const at::Tensor & eigvecs) {
    // d / dc * d / dq * H in diabatic representation
    at::Tensor Dc00DqHd, Dc01DqHd, Dc11DqHd;
    std::tie(Dc00DqHd, Dc01DqHd, Dc11DqHd) = libHd::analytical_DcDqHd(c00, c01, c11, q);
    // (d / dq * H)c
    at::Tensor DqHd = libHd::analytical_DqHd(c00, c01, c11, q);
    at::Tensor DqH_c = tchem::linalg::UT_sy_U(DqHd, eigvecs);
    // nac
    at::Tensor Dc00O = tchem::linalg::sy4matmvmulsy3(Dc00DqHd.transpose(-1, -2), DqHd),
               Dc01O = tchem::linalg::sy4matmvmulsy3(Dc01DqHd.transpose(-1, -2), DqHd),
               Dc11O = tchem::linalg::sy4matmvmulsy3(Dc11DqHd.transpose(-1, -2), DqHd);
    Dc00O = Dc00O + Dc00O.transpose(0, 1);
    Dc01O = Dc01O + Dc01O.transpose(0, 1);
    Dc11O = Dc11O + Dc11O.transpose(0, 1);
    at::Tensor nac_c00 = tchem::linalg::UT_sy_U(Dc00O, eigvecs),
               nac_c01 = tchem::linalg::UT_sy_U(Dc01O, eigvecs),
               nac_c11 = tchem::linalg::UT_sy_U(Dc11O, eigvecs);
    nac_c00[0][1] /= eigvals[1] - eigvals[0];
    nac_c01[0][1] /= eigvals[1] - eigvals[0];
    nac_c11[0][1] /= eigvals[1] - eigvals[0];
    // d / dc * (dH / dq)c
    at::Tensor Dc00_DqH_c = tchem::linalg::UT_sy_U(Dc00DqHd, eigvecs)
                          + commutor(DqH_c, nac_c00),
               Dc01_DqH_c = tchem::linalg::UT_sy_U(Dc01DqHd, eigvecs)
                          + commutor(DqH_c, nac_c01),
               Dc11_DqH_c = tchem::linalg::UT_sy_U(Dc11DqHd, eigvecs)
                          + commutor(DqH_c, nac_c11);
    return std::make_tuple(Dc00_DqH_c, Dc01_DqH_c, Dc11_DqH_c);
}

// Numerical version
std::tuple<at::Tensor, at::Tensor, at::Tensor> numerical_Dc_DqH_c
(const at::Tensor & c00, const at::Tensor & c01, const at::Tensor & c11, const at::Tensor & q,
const at::Tensor & DqH_c) {
    tchem::chem::Phaser phaser(2);
    at::Tensor Dc00_DqH_c = q.new_zeros({2, 2, 2, 3});
    for (size_t l = 0; l < 3; l++) {
        at::Tensor c00_plus = c00.clone();
        c00_plus[l] += 1e-5;
        at::Tensor DqH_c_plus = libHd::DqH_c(c00_plus, c01, c11, q);
        phaser.fix_ob_(DqH_c_plus, DqH_c);
        at::Tensor c00_minus = c00.clone();
        c00_minus[l] -= 1e-5;
        at::Tensor DqH_c_minus = libHd::DqH_c(c00_minus, c01, c11, q);
        phaser.fix_ob_(DqH_c_minus, DqH_c);
        for (size_t i = 0; i < 2; i++)
        for (size_t j = i; j < 2; j++)
        for (size_t k = 0; k < 2; k++)
        Dc00_DqH_c[i][j][k][l] = (DqH_c_plus[i][j][k] - DqH_c_minus[i][j][k]) / 2e-5;
    }
    at::Tensor Dc01_DqH_c = q.new_zeros({2, 2, 2, 2});
    for (size_t l = 0; l < 2; l++) {
        at::Tensor c01_plus = c01.clone();
        c01_plus[l] += 1e-5;
        at::Tensor DqH_c_plus = libHd::DqH_c(c00, c01_plus, c11, q);
        phaser.fix_ob_(DqH_c_plus, DqH_c);
        at::Tensor c01_minus = c01.clone();
        c01_minus[l] -= 1e-5;
        at::Tensor DqH_c_minus = libHd::DqH_c(c00, c01_minus, c11, q);
        phaser.fix_ob_(DqH_c_minus, DqH_c);
        for (size_t i = 0; i < 2; i++)
        for (size_t j = i; j < 2; j++)
        for (size_t k = 0; k < 2; k++)
        Dc01_DqH_c[i][j][k][l] = (DqH_c_plus[i][j][k] - DqH_c_minus[i][j][k]) / 2e-5;
    }
    at::Tensor Dc11_DqH_c = q.new_zeros({2, 2, 2, 3});
    for (size_t l = 0; l < 3; l++) {
        at::Tensor c11_plus = c11.clone();
        c11_plus[l] += 1e-5;
        at::Tensor DqH_c_plus = libHd::DqH_c(c00, c01, c11_plus, q);
        phaser.fix_ob_(DqH_c_plus, DqH_c);
        at::Tensor c11_minus = c11.clone();
        c11_minus[l] -= 1e-5;
        at::Tensor DqH_c_minus = libHd::DqH_c(c00, c01, c11_minus, q);
        phaser.fix_ob_(DqH_c_minus, DqH_c);
        for (size_t i = 0; i < 2; i++)
        for (size_t j = i; j < 2; j++)
        for (size_t k = 0; k < 2; k++)
        Dc11_DqH_c[i][j][k][l] = (DqH_c_plus[i][j][k] - DqH_c_minus[i][j][k]) / 2e-5;
    }
    return std::make_tuple(Dc00_DqH_c, Dc01_DqH_c, Dc11_DqH_c);
}

void test_Dc_DqH_c() {
    c10::TensorOptions top = at::TensorOptions().dtype(torch::kFloat64);
    at::Tensor c00 = at::rand(3, top),
               c01 = at::rand(2, top),
               c11 = at::rand(3, top);
    c00.set_requires_grad(true);
    c01.set_requires_grad(true);
    c11.set_requires_grad(true);
    at::Tensor q = at::rand(2, top);
    q.set_requires_grad(true);
    at::Tensor Dc00_DqH_c, Dc01_DqH_c, Dc11_DqH_c, DqH_c, eigvals, eigvecs;
    std::tie(Dc00_DqH_c, Dc01_DqH_c, Dc11_DqH_c, DqH_c, eigvals, eigvecs) = Dc_DqH_c(c00, c01, c11, q);
    at::Tensor Dc00_DqH_c_A, Dc01_DqH_c_A, Dc11_DqH_c_A;
    std::tie(Dc00_DqH_c_A, Dc01_DqH_c_A, Dc11_DqH_c_A) = analytical_Dc_DqH_c(c00, c01, c11, q, eigvals, eigvecs);
    at::Tensor Dc00_DqH_c_N, Dc01_DqH_c_N, Dc11_DqH_c_N;
    std::tie(Dc00_DqH_c_N, Dc01_DqH_c_N, Dc11_DqH_c_N) = numerical_Dc_DqH_c(c00, c01, c11, q, DqH_c);
    std::cout << "\nd / dc * (d / dq * H)c:\n"
              << "autograd vs numerical: "
              << (Dc00_DqH_c - Dc00_DqH_c_N).norm().item<double>() << "    "
              << (Dc01_DqH_c - Dc01_DqH_c_N).norm().item<double>() << "    "
              << (Dc11_DqH_c - Dc11_DqH_c_N).norm().item<double>() << '\n'
              << "analytical vs numerical: "
              << (Dc00_DqH_c_A - Dc00_DqH_c_N).norm().item<double>() << "    "
              << (Dc01_DqH_c_A - Dc01_DqH_c_N).norm().item<double>() << "    "
              << (Dc11_DqH_c_A - Dc11_DqH_c_N).norm().item<double>() << '\n';
}