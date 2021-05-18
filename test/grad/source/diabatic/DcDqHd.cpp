#include "../Hd.hpp"

void test_DcDqHd() {
  c10::TensorOptions top = at::TensorOptions().dtype(torch::kFloat64);
    at::Tensor c00 = at::rand(3, top),
               c01 = at::rand(2, top),
               c11 = at::rand(3, top);
    c00.set_requires_grad(true);
    c01.set_requires_grad(true);
    c11.set_requires_grad(true);
    at::Tensor q = at::rand(2, top);
    q.set_requires_grad(true);
    at::Tensor ddHd_c00, ddHd_c00_A,
               ddHd_c01, ddHd_c01_A,
               ddHd_c11, ddHd_c11_A;
    std::tie(ddHd_c00  , ddHd_c01  , ddHd_c11  ) = libHd::           DcDqHd(c00, c01, c11, q);
    std::tie(ddHd_c00_A, ddHd_c01_A, ddHd_c11_A) = libHd::analytical_DcDqHd(c00, c01, c11, q);
    std::cout << "\nd / dc * d / dq * Hd: "
              << ((ddHd_c00 - ddHd_c00_A).norm()
                + (ddHd_c01 - ddHd_c01_A).norm()
                + (ddHd_c11 - ddHd_c11_A).norm()
                 ).item<double>()
              << '\n';
}