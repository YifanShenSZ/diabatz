#include "../Hd.hpp"

void test_DcHd() {
  c10::TensorOptions top = at::TensorOptions().dtype(torch::kFloat64);
    at::Tensor c00 = at::rand(3, top),
               c01 = at::rand(2, top),
               c11 = at::rand(3, top);
    c00.set_requires_grad(true);
    c01.set_requires_grad(true);
    c11.set_requires_grad(true);
    at::Tensor q = at::rand(2, top);
    q.set_requires_grad(true);
    at::Tensor Dc00Hc, Dc00Hc_A,
               Dc01Hc, Dc01Hc_A,
               Dc11Hc, Dc11Hc_A;
    std::tie(Dc00Hc  , Dc01Hc  , Dc11Hc  ) = libHd::           DcHd(c00, c01, c11, q);
    std::tie(Dc00Hc_A, Dc01Hc_A, Dc11Hc_A) = libHd::analytical_DcHd(c00, c01, c11, q);
    std::cout << "\nd / dc * Hd: "
              << ((Dc00Hc - Dc00Hc_A).norm()
                + (Dc01Hc - Dc01Hc_A).norm()
                + (Dc11Hc - Dc11Hc_A).norm()
                 ).item<double>()
              << '\n';
}