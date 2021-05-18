#include <tchem/linalg.hpp>
#include <tchem/chemistry.hpp>

#include "../Hd.hpp"
#include "../commutor.hpp"

// Return d / dc of Hc matrix elements,
// Hc matrix, eigenvalues and eigenvectors of dHd / dq . dHd / dq
std::tuple<at::Tensor, at::Tensor, at::Tensor,
at::Tensor, at::Tensor, at::Tensor> DcHc
(const at::Tensor & c00, const at::Tensor & c01, const at::Tensor & c11, const at::Tensor & q) {
    assert(("c00 must require gradient", c00.requires_grad()));
    assert(("c01 must require gradient", c01.requires_grad()));
    assert(("c11 must require gradient", c11.requires_grad()));
    assert(("q must require gradient", q.requires_grad()));
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
    at::Tensor Hc = tchem::linalg::UT_sy_U(Hd, eigvecs);
    at::Tensor Dc00Hc = q.new_zeros({2, 2, 3}),
               Dc01Hc = q.new_zeros({2, 2, 2}),
               Dc11Hc = q.new_zeros({2, 2, 3});
    for (size_t i = 0; i < 2; i++)
    for (size_t j = i; j < 2; j++) {
        torch::autograd::variable_list g = torch::autograd::grad({Hc[i][j]}, {c00, c01, c11}, {}, true);
        Dc00Hc[i][j].copy_(g[0]);
        Dc01Hc[i][j].copy_(g[1]);
        Dc11Hc[i][j].copy_(g[2]);
    }
    return std::make_tuple(Dc00Hc, Dc01Hc, Dc11Hc, Hc, eigvals, eigvecs);
}

// Analytical version
std::tuple<at::Tensor, at::Tensor, at::Tensor> analytical_DcHc
(const at::Tensor & c00, const at::Tensor & c01, const at::Tensor & c11, const at::Tensor & q,
const at::Tensor & eigvals, const at::Tensor & eigvecs) {
    // d / dc * H in diabatic representation
    at::Tensor Dc00Hd, Dc01Hd, Dc11Hd;
    std::tie(Dc00Hd, Dc01Hd, Dc11Hd) = libHd::analytical_DcHd(c00, c01, c11, q);
    // H in composite representation
    at::Tensor Hd = libHd::Hd(c00, c01, c11, q);
    at::Tensor Hc = tchem::linalg::UT_sy_U(Hd, eigvecs);
    Hc[1][0].copy_(Hc[0][1]); // For gematoutermul
    // nac
    at::Tensor DqHd = libHd::analytical_DqHd(c00, c01, c11, q);
    at::Tensor Dc00DqHd, Dc01DqHd, Dc11DqHd;
    std::tie(Dc00DqHd, Dc01DqHd, Dc11DqHd) = libHd::analytical_DcDqHd(c00, c01, c11, q);
    at::Tensor Dc00O = tchem::linalg::sy4matmvmulsy3(Dc00DqHd.transpose_(-1, -2), DqHd),
               Dc01O = tchem::linalg::sy4matmvmulsy3(Dc01DqHd.transpose_(-1, -2), DqHd),
               Dc11O = tchem::linalg::sy4matmvmulsy3(Dc11DqHd.transpose_(-1, -2), DqHd);
    Dc00O = Dc00O + Dc00O.transpose(0, 1);
    Dc01O = Dc01O + Dc01O.transpose(0, 1);
    Dc11O = Dc11O + Dc11O.transpose(0, 1);
    at::Tensor nac_c00 = tchem::linalg::UT_sy_U(Dc00O, eigvecs),
               nac_c01 = tchem::linalg::UT_sy_U(Dc01O, eigvecs),
               nac_c11 = tchem::linalg::UT_sy_U(Dc11O, eigvecs);
    nac_c00[0][1] /= eigvals[1] - eigvals[0];
    nac_c01[0][1] /= eigvals[1] - eigvals[0];
    nac_c11[0][1] /= eigvals[1] - eigvals[0];
    // Combine to d / dq * Hc
    at::Tensor Dc00Hc = tchem::linalg::UT_sy_U(Dc00Hd, eigvecs)
                      + commutor(Hc, nac_c00),
               Dc01Hc = tchem::linalg::UT_sy_U(Dc01Hd, eigvecs)
                      + commutor(Hc, nac_c01),
               Dc11Hc = tchem::linalg::UT_sy_U(Dc11Hd, eigvecs)
                      + commutor(Hc, nac_c11);
    return std::make_tuple(Dc00Hc, Dc01Hc, Dc11Hc);
}

// Numerical version
std::tuple<at::Tensor, at::Tensor, at::Tensor> numerical_DcHc
(const at::Tensor & c00, const at::Tensor & c01, const at::Tensor & c11, const at::Tensor & q,
const at::Tensor & Hc) {
    tchem::chem::Phaser phaser(2);
    at::Tensor Dc00Hc = q.new_zeros({2, 2, 3});
    for (size_t i = 0; i < 3; i++) {
        at::Tensor c00_p = c00.clone();
        c00_p[i] += 1e-5;
        at::Tensor Hc_p = libHd::Hc(c00_p, c01, c11, q);
        phaser.fix_ob_(Hc_p, Hc);
        at::Tensor c00_m = c00.clone();
        c00_m[i] -= 1e-5;
        at::Tensor Hc_m = libHd::Hc(c00_m, c01, c11, q);
        phaser.fix_ob_(Hc_m, Hc);
        for (size_t istate = 0; istate < 2; istate++)
        for (size_t jstate = istate; jstate < 2; jstate++)
        Dc00Hc[istate][jstate][i] = (Hc_p[istate][jstate] - Hc_m[istate][jstate]) / 2e-5;
    }
    at::Tensor Dc01Hc = q.new_zeros({2, 2, 2});
    for (size_t i = 0; i < 2; i++) {
        at::Tensor c01_p = c01.clone();
        c01_p[i] += 1e-5;
        at::Tensor Hc_p = libHd::Hc(c00, c01_p, c11, q);
        phaser.fix_ob_(Hc_p, Hc);
        at::Tensor c01_m = c01.clone();
        c01_m[i] -= 1e-5;
        at::Tensor Hc_m = libHd::Hc(c00, c01_m, c11, q);
        phaser.fix_ob_(Hc_m, Hc);
        for (size_t istate = 0; istate < 2; istate++)
        for (size_t jstate = istate; jstate < 2; jstate++)
        Dc01Hc[istate][jstate][i] = (Hc_p[istate][jstate] - Hc_m[istate][jstate]) / 2e-5;
    }
    at::Tensor Dc11Hc = q.new_zeros({2, 2, 3});
    for (size_t i = 0; i < 3; i++) {
        at::Tensor c11_p = c11.clone();
        c11_p[i] += 1e-5;
        at::Tensor Hc_p = libHd::Hc(c00, c01, c11_p, q);
        phaser.fix_ob_(Hc_p, Hc);
        at::Tensor c11_m = c11.clone();
        c11_m[i] -= 1e-5;
        at::Tensor Hc_m = libHd::Hc(c00, c01, c11_m, q);
        phaser.fix_ob_(Hc_m, Hc);
        for (size_t istate = 0; istate < 2; istate++)
        for (size_t jstate = istate; jstate < 2; jstate++)
        Dc11Hc[istate][jstate][i] = (Hc_p[istate][jstate] - Hc_m[istate][jstate]) / 2e-5;
    }
    return std::make_tuple(Dc00Hc, Dc01Hc, Dc11Hc);
}

void test_DcHc() {
    c10::TensorOptions top = at::TensorOptions().dtype(torch::kFloat64);
    at::Tensor c00 = at::rand(3, top),
               c01 = at::rand(2, top),
               c11 = at::rand(3, top);
    c00.set_requires_grad(true);
    c01.set_requires_grad(true);
    c11.set_requires_grad(true);
    at::Tensor q = at::rand(2, top);
    q.set_requires_grad(true);
    at::Tensor Dc00Hc, Dc01Hc, Dc11Hc, Hc, eigvals, eigvecs;
    std::tie(Dc00Hc, Dc01Hc, Dc11Hc, Hc, eigvals, eigvecs) = DcHc(c00, c01, c11, q);
    c00.set_requires_grad(false);
    c01.set_requires_grad(false);
    c11.set_requires_grad(false);
    q.set_requires_grad(false);
    at::Tensor Dc00Hc_A, Dc01Hc_A, Dc11Hc_A;
    std::tie(Dc00Hc_A, Dc01Hc_A, Dc11Hc_A) = analytical_DcHc(c00, c01, c11, q, eigvals, eigvecs);
    at::Tensor Dc00Hc_N, Dc01Hc_N, Dc11Hc_N;
    std::tie(Dc00Hc_N, Dc01Hc_N, Dc11Hc_N) =  numerical_DcHc(c00, c01, c11, q, Hc);
    std::cout << "\nd / dc * Hc:\n"
              << "autograd vs numerical: "
              << (Dc00Hc - Dc00Hc_N).norm().item<double>() << "    "
              << (Dc01Hc - Dc01Hc_N).norm().item<double>() << "    "
              << (Dc11Hc - Dc11Hc_N).norm().item<double>() << '\n'
              << "analytical vs numerical: "
              << (Dc00Hc_A - Dc00Hc_N).norm().item<double>() << "    "
              << (Dc01Hc_A - Dc01Hc_N).norm().item<double>() << "    "
              << (Dc11Hc_A - Dc11Hc_N).norm().item<double>() << '\n';
}