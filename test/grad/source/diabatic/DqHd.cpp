#include "../Hd.hpp"

void test_DqHd() {
    c10::TensorOptions top = at::TensorOptions().dtype(torch::kFloat64);
    at::Tensor c00 = at::rand(3, top),
               c01 = at::rand(2, top),
               c11 = at::rand(3, top);
    c00.set_requires_grad(true);
    c01.set_requires_grad(true);
    c11.set_requires_grad(true);
    at::Tensor q = at::rand(2, top);
    q.set_requires_grad(true);
    std::cout << "\nd / dq * Hd: "
              << (libHd::DqHd(c00, c01, c11, q)
                - libHd::analytical_DqHd(c00, c01, c11, q)
                 ).norm().item<double>()
              << '\n';
}