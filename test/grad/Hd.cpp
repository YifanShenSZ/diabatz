#include <tchem/linalg.hpp>

namespace libHd {

at::Tensor Hd(const at::Tensor & c00, const at::Tensor & c01, const at::Tensor & c11, const at::Tensor & q) {
    at::Tensor Hd = q.new_empty({2, 2});
    Hd[0][0] = c00[0] * q[0] * q[0] + c00[1] * q[0] * q[1] + c00[2] * q[1] * q[1];
    Hd[0][1] = c01[0] * q[0] + c01[1] * q[1];
    Hd[1][1] = c11[0] * q[0] * q[0] + c11[1] * q[0] * q[1] + c11[2] * q[1] * q[1];
    return Hd;
}

// d / dq of Hd matrix elements
// Because of diabaticity, this is equivalent to
// the matrix form of operator d / dq * H in diabatic representation
at::Tensor DqHd(const at::Tensor & c00, const at::Tensor & c01, const at::Tensor & c11, const at::Tensor & q) {
    assert(("q must require gradient", q.requires_grad()));
    at::Tensor Hd = q.new_empty({2, 2});
    Hd[0][0] = c00[0] * q[0] * q[0] + c00[1] * q[0] * q[1] + c00[2] * q[1] * q[1];
    Hd[0][1] = c01[0] * q[0] + c01[1] * q[1];
    Hd[1][1] = c11[0] * q[0] * q[0] + c11[1] * q[0] * q[1] + c11[2] * q[1] * q[1];
    at::Tensor dHd = q.new_zeros({2, 2, 2});
    for (size_t i = 0; i < 2; i++)
    for (size_t j = i; j < 2; j++) {
        torch::autograd::variable_list g = torch::autograd::grad({Hd[i][j]}, {q}, {}, true, true);
        dHd[i][j] = g[0];
    }
    return dHd;
}
// Analytical version
at::Tensor analytical_DqHd(const at::Tensor & c00, const at::Tensor & c01, const at::Tensor & c11, const at::Tensor & q) {
    at::Tensor dHd = q.new_zeros({2, 2, 2});
    dHd[0][0][0] = 2.0 * c00[0] * q[0] + c00[1] * q[1];
    dHd[0][0][1] = 2.0 * c00[2] * q[1] + c00[1] * q[0];
    dHd[0][1] = c01;
    dHd[1][1][0] = 2.0 * c11[0] * q[0] + c11[1] * q[1];
    dHd[1][1][1] = 2.0 * c11[2] * q[1] + c11[1] * q[0];
    return dHd;
}

// d / dc of Hd matrix elements
// Because of diabaticity, this is equivalent to
// the matrix form of operator d / dc * H in diabatic representation
std::tuple<at::Tensor, at::Tensor, at::Tensor> DcHd
(const at::Tensor & c00, const at::Tensor & c01, const at::Tensor & c11, const at::Tensor & q) {
    assert(("c00 must require gradient", c00.requires_grad()));
    assert(("c01 must require gradient", c01.requires_grad()));
    assert(("c11 must require gradient", c11.requires_grad()));
    at::Tensor Hd = q.new_empty({2, 2});
    Hd[0][0] = c00[0] * q[0] * q[0] + c00[1] * q[0] * q[1] + c00[2] * q[1] * q[1];
    Hd[0][1] = c01[0] * q[0] + c01[1] * q[1];
    Hd[1][1] = c11[0] * q[0] * q[0] + c11[1] * q[0] * q[1] + c11[2] * q[1] * q[1];
    at::Tensor dHd_c00 = q.new_zeros({2, 2, 3}),
               dHd_c01 = q.new_zeros({2, 2, 2}),
               dHd_c11 = q.new_zeros({2, 2, 3});
    for (size_t i = 0; i < 2; i++)
    for (size_t j = i; j < 2; j++) {
        torch::autograd::variable_list g = torch::autograd::grad({Hd[i][j]}, {c00, c01, c11}, {}, true, true);
        dHd_c00[i][j] = g[0];
        dHd_c01[i][j] = g[1];
        dHd_c11[i][j] = g[2];
    }
    return std::make_tuple(dHd_c00, dHd_c01, dHd_c11);
}
// Analytical version
std::tuple<at::Tensor, at::Tensor, at::Tensor> analytical_DcHd
(const at::Tensor & c00, const at::Tensor & c01, const at::Tensor & c11, const at::Tensor & q) {
    at::Tensor dHd_c00 = q.new_zeros({2, 2, 3}),
               dHd_c01 = q.new_zeros({2, 2, 2}),
               dHd_c11 = q.new_zeros({2, 2, 3});
    dHd_c00[0][0][0] = q[0] * q[0];
    dHd_c00[0][0][1] = q[0] * q[1];
    dHd_c00[0][0][2] = q[1] * q[1];
    dHd_c01[0][1][0] = q[0];
    dHd_c01[0][1][1] = q[1];
    dHd_c11[1][1][0] = q[0] * q[0];
    dHd_c11[1][1][1] = q[0] * q[1];
    dHd_c11[1][1][2] = q[1] * q[1];
    return std::make_tuple(dHd_c00, dHd_c01, dHd_c11);
}

// d / dc * d / dq of Hd matrix elements
// Because of diabaticity, this is equivalent to
// the matrix form of operator d / dc * d / dq * H in diabatic representation
std::tuple<at::Tensor, at::Tensor, at::Tensor> DcDqHd
(const at::Tensor & c00, const at::Tensor & c01, const at::Tensor & c11, const at::Tensor & q) {
    assert(("c00 must require gradient", c00.requires_grad()));
    assert(("c01 must require gradient", c01.requires_grad()));
    assert(("c11 must require gradient", c11.requires_grad()));
    assert(("q must require gradient", q.requires_grad()));
    // Hd
    at::Tensor Hd = q.new_empty({2, 2});
    Hd[0][0] = c00[0] * q[0] * q[0] + c00[1] * q[0] * q[1] + c00[2] * q[1] * q[1];
    Hd[0][1] = c01[0] * q[0] + c01[1] * q[1];
    Hd[1][1] = c11[0] * q[0] * q[0] + c11[1] * q[0] * q[1] + c11[2] * q[1] * q[1];
    // dHd / dq
    at::Tensor dHd = q.new_zeros({2, 2, 2});
    for (size_t i = 0; i < 2; i++)
    for (size_t j = i; j < 2; j++) {
        torch::autograd::variable_list g = torch::autograd::grad({Hd[i][j]}, {q}, {}, true, true);
        dHd[i][j] = g[0];
    }
    // ddHd / dq / dc
    at::Tensor ddHd_c00 = q.new_zeros({2, 2, 2, 3}),
               ddHd_c01 = q.new_zeros({2, 2, 2, 2}),
               ddHd_c11 = q.new_zeros({2, 2, 2, 3});
    for (size_t i = 0; i < 2; i++)
    for (size_t j = i; j < 2; j++)
    for (size_t k = 0; k < 2; k++) {
        torch::autograd::variable_list g = torch::autograd::grad({dHd[i][j][k]}, {c00, c01, c11}, {}, true);
        ddHd_c00[i][j][k] = g[0];
        ddHd_c01[i][j][k] = g[1];
        ddHd_c11[i][j][k] = g[2];
    }
    return std::make_tuple(ddHd_c00, ddHd_c01, ddHd_c11);
}
// Analytical version
std::tuple<at::Tensor, at::Tensor, at::Tensor> analytical_DcDqHd
(const at::Tensor & c00, const at::Tensor & c01, const at::Tensor & c11, const at::Tensor & q) {
    at::Tensor ddHd_c00 = q.new_zeros({2, 2, 2, 3}),
               ddHd_c01 = q.new_zeros({2, 2, 2, 2}),
               ddHd_c11 = q.new_zeros({2, 2, 2, 3});
    ddHd_c00[0][0][0][0] = 2.0 * q[0];
    ddHd_c00[0][0][0][1] = q[1];
    ddHd_c00[0][0][1][2] = 2.0 * q[1];
    ddHd_c00[0][0][1][1] = q[0];
    ddHd_c01[0][1][0][0] = 1.0;
    ddHd_c01[0][1][1][1] = 1.0;
    ddHd_c11[1][1][0][0] = 2.0 * q[0];
    ddHd_c11[1][1][0][1] = q[1];
    ddHd_c11[1][1][1][2] = 2.0 * q[1];
    ddHd_c11[1][1][1][1] = q[0];
    return std::make_tuple(ddHd_c00, ddHd_c01, ddHd_c11);
}





at::Tensor energies(const at::Tensor & c00, const at::Tensor & c01, const at::Tensor & c11, const at::Tensor & q) {
    at::Tensor H = Hd(c00, c01, c11, q);
    at::Tensor energies, states;
    std::tie(energies, states) = H.symeig();
    return energies;
}

// The matrix form of operator d / dq * H in adiabatic representation
at::Tensor DqH_a(const at::Tensor & c00, const at::Tensor & c01, const at::Tensor & c11, const at::Tensor & q) {
    // d / dq * H in diabatic representation
    // Because of diabaticity it is equivalent to d / dq * Hd
    at::Tensor dH = analytical_DqHd(c00, c01, c11, q);
    // Transform to adiabatic representation
    at::Tensor H = Hd(c00, c01, c11, q);
    at::Tensor energies, states;
    std::tie(energies, states) = H.symeig(true);
    at::Tensor dHa = tchem::LA::UT_sy_U(dH, states);
    return dHa;
}





at::Tensor Hc(const at::Tensor & c00, const at::Tensor & c01, const at::Tensor & c11, const at::Tensor & q) {
    at::Tensor H = Hd(c00, c01, c11, q);
    at::Tensor dH = analytical_DqHd(c00, c01, c11, q);
    at::Tensor dHdH = tchem::LA::sy3matdotmul(dH, dH);
    at::Tensor eigvals, eigvecs;
    std::tie(eigvals, eigvecs) = dHdH.symeig(true);
    at::Tensor Hc = tchem::LA::UT_sy_U(H, eigvecs);
    return Hc;
}

// The matrix form of operator d / dq * H in composite representation
at::Tensor DqH_c(const at::Tensor & c00, const at::Tensor & c01, const at::Tensor & c11, const at::Tensor & q) {
    // d / dq * H in diabatic representation
    // Because of diabaticity it is equivalent to d / dq * Hd
    at::Tensor DqHd = analytical_DqHd(c00, c01, c11, q);
    // Transform to composite representation
    at::Tensor DqHdDqHd = tchem::LA::sy3matdotmul(DqHd, DqHd);
    at::Tensor eigvals, eigvecs;
    std::tie(eigvals, eigvecs) = DqHdDqHd.symeig(true);
    at::Tensor DqH_c = tchem::LA::UT_sy_U(DqHd, eigvecs);
    return DqH_c;
}

} // namespace libHd