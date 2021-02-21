#include <tchem/linalg.hpp>

#include "Hd.hpp"

std::tuple<at::Tensor, at::Tensor, at::Tensor> analytical
(const at::Tensor & c00, const at::Tensor & c01, const at::Tensor & c11, const at::Tensor & q) {
    at::Tensor DqHd = libHd::analytical_DqHd(c00, c01, c11, q);
    at::Tensor Dc00DqHd, Dc01DqHd, Dc11DqHd;
    std::tie(Dc00DqHd, Dc01DqHd, Dc11DqHd) = libHd::analytical_DcDqHd(c00, c01, c11, q);
    at::Tensor Dc00O = tchem::linalg::sy4matmvmulsy3(Dc00DqHd.transpose_(-1, -2), DqHd),
               Dc01O = tchem::linalg::sy4matmvmulsy3(Dc01DqHd.transpose_(-1, -2), DqHd),
               Dc11O = tchem::linalg::sy4matmvmulsy3(Dc11DqHd.transpose_(-1, -2), DqHd);
    Dc00O = Dc00O + Dc00O.transpose(0, 1);
    Dc01O = Dc01O + Dc01O.transpose(0, 1);
    Dc11O = Dc11O + Dc11O.transpose(0, 1);
    return std::make_tuple(Dc00O, Dc01O, Dc11O);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> numerical
(const at::Tensor & c00, const at::Tensor & c01, const at::Tensor & c11, const at::Tensor & q) {
    at::Tensor Dc00O = q.new_zeros({2, 2, 3});
    for (size_t i = 0; i < 3; i++) {
        at::Tensor c00_p = c00.clone();
        c00_p[i] += 1e-5;
        at::Tensor dH_p = libHd::analytical_DqHd(c00_p, c01, c11, q);
        at::Tensor O_p = tchem::linalg::sy3matdotmul(dH_p, dH_p);
        at::Tensor c00_m = c00.clone();
        c00_m[i] -= 1e-5;
        at::Tensor dH_m = libHd::analytical_DqHd(c00_m, c01, c11, q);
        at::Tensor O_m = tchem::linalg::sy3matdotmul(dH_m, dH_m);
        for (size_t istate = 0; istate < 2; istate++)
        for (size_t jstate = 0; jstate < 2; jstate++)
        Dc00O[istate][jstate][i] = (O_p[istate][jstate] - O_m[istate][jstate]) / 2e-5;
    }
    at::Tensor Dc01O = q.new_zeros({2, 2, 2});
    for (size_t i = 0; i < 2; i++) {
        at::Tensor c01_p = c01.clone();
        c01_p[i] += 1e-5;
        at::Tensor dH_p = libHd::analytical_DqHd(c00, c01_p, c11, q);
        at::Tensor O_p = tchem::linalg::sy3matdotmul(dH_p, dH_p);
        at::Tensor c01_m = c01.clone();
        c01_m[i] -= 1e-5;
        at::Tensor dH_m = libHd::analytical_DqHd(c00, c01_m, c11, q);
        at::Tensor O_m = tchem::linalg::sy3matdotmul(dH_m, dH_m);
        for (size_t istate = 0; istate < 2; istate++)
        for (size_t jstate = 0; jstate < 2; jstate++)
        Dc01O[istate][jstate][i] = (O_p[istate][jstate] - O_m[istate][jstate]) / 2e-5;
    }
    at::Tensor Dc11O = q.new_zeros({2, 2, 3});
    for (size_t i = 0; i < 3; i++) {
        at::Tensor c11_p = c11.clone();
        c11_p[i] += 1e-5;
        at::Tensor dH_p = libHd::analytical_DqHd(c00, c01, c11_p, q);
        at::Tensor O_p = tchem::linalg::sy3matdotmul(dH_p, dH_p);
        at::Tensor c11_m = c11.clone();
        c11_m[i] -= 1e-5;
        at::Tensor dH_m = libHd::analytical_DqHd(c00, c01, c11_m, q);
        at::Tensor O_m = tchem::linalg::sy3matdotmul(dH_m, dH_m);
        for (size_t istate = 0; istate < 2; istate++)
        for (size_t jstate = 0; jstate < 2; jstate++)
        Dc11O[istate][jstate][i] = (O_p[istate][jstate] - O_m[istate][jstate]) / 2e-5;
    }
    return std::make_tuple(Dc00O, Dc01O, Dc11O);
}

void test_composite() {
    c10::TensorOptions top = at::TensorOptions().dtype(torch::kFloat64);
    at::Tensor c00 = at::rand(3, top),
               c01 = at::rand(2, top),
               c11 = at::rand(3, top);
    at::Tensor q = at::rand(2, top);
    at::Tensor Dc00O_A, Dc01O_A, Dc11O_A;
    std::tie(Dc00O_A, Dc01O_A, Dc11O_A) = analytical(c00, c01, c11, q);
    at::Tensor Dc00O_N, Dc01O_N, Dc11O_N;
    std::tie(Dc00O_N, Dc01O_N, Dc11O_N) =  numerical(c00, c01, c11, q);
    std::cout << "\nd / dc * (d / dq * Hd . d / dq * Hd):\n"
              << (Dc00O_A - Dc00O_N).norm().item<double>() << "    "
              << (Dc01O_A - Dc01O_N).norm().item<double>() << "    "
              << (Dc11O_A - Dc11O_N).norm().item<double>() << '\n';
}