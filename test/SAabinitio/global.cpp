#include "global.hpp"

std::shared_ptr<tchem::IC::SASICSet> sasicset;

// Given Cartesian coordinate r,
// return symmetry adapted internal coordinates
// and their Jacobians over r
std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> cart2int(const at::Tensor & r) {
    assert(("Define symmetry adaptated and scaled internal coordinate before use", sasicset));
    // Cartesian coordinate -> internal coordinate
    at::Tensor q, J;
    std::tie(q, J) = sasicset->compute_IC_J(r);
    q.set_requires_grad(true);
    // internal coordinate -> symmetry adapted internal coordinate
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
    return std::make_tuple(qs, Js);
}