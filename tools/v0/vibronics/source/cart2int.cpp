#include "../include/cart2int.hpp"

std::shared_ptr<tchem::IC::SASICSet> sasicset;

// Given Cartesian coordinate r,
// return CNPI group symmetry adapted internal coordinates and corresponding Jacobians
std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> cart2int(const at::Tensor & r) {
    assert(("Define CNPI group symmetry adaptated and scaled internal coordinate before use", sasicset));
    // Cartesian coordinate -> internal coordinate
    at::Tensor q, J;
    std::tie(q, J) = sasicset->compute_IC_J(r);
    q.set_requires_grad(true);
    // internal coordinate -> CNPI group symmetry adapted internal coordinate
    std::vector<at::Tensor> qs = (*sasicset)(q);
    std::vector<at::Tensor> Js = std::vector<at::Tensor>(qs.size());
    for (size_t i = 0; i < qs.size(); i++) {
        Js[i] = qs[i].new_empty({qs[i].size(0), q.size(0)});
        for (size_t j = 0; j < qs[i].size(0); j++) {
            std::vector<at::Tensor> g = torch::autograd::grad({qs[i][j]}, {q}, {}, true);
            Js[i][j].copy_(g[0]);
        }
        Js[i] = Js[i].mm(J);
    }
    // Free autograd graph
    for (at::Tensor & q : qs) q.detach_();
    return std::make_tuple(qs, Js);
}

at::Tensor dHd_cart2int(const at::Tensor & r, const at::Tensor & cartdHd) {
    std::vector<at::Tensor> qs, Js;
    std::tie(qs, Js) = cart2int(r);
    at::Tensor q = at::cat(qs), J = at::cat(Js);
    at::Tensor JJT = J.mm(J.transpose(0, 1));
    at::Tensor cholesky = JJT.cholesky();
    at::Tensor inverse = at::cholesky_inverse(cholesky);
    at::Tensor mat4cart2int = inverse.mm(J);
    at::Tensor intdHd = r.new_empty({cartdHd.size(0), cartdHd.size(1), q.size(0)});
    for (size_t i = 0; i < intdHd.size(0); i++)
    for (size_t j = i; j < intdHd.size(1); j++)
    intdHd[i][j].copy_(mat4cart2int.mv(cartdHd[i][j]));
    return intdHd;
}