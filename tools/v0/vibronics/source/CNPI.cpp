#include "../include/CNPI.hpp"

std::shared_ptr<tchem::IC::SASICSet> sasicset;

// given Cartesian coordinate r,
// return CNPI group symmetry adapted internal coordinates and corresponding Jacobians
std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> cart2CNPI(const at::Tensor & r) {
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

// concatenate CNPI group symmetry adapted tensors to point group symmetry adapted tensors
std::vector<at::Tensor> cat(const std::vector<at::Tensor> & xs, const std::vector<std::vector<size_t>> & point2CNPI) {
    size_t n_point_irreds = point2CNPI.size();
    std::vector<at::Tensor> ys(n_point_irreds);
    for (size_t i = 0; i < n_point_irreds; i++) {
        std::vector<at::Tensor> xmatches(point2CNPI[i].size());
        for (size_t j = 0; j < point2CNPI[i].size(); j++) xmatches[j] = xs[point2CNPI[i][j]];
        ys[i] = at::cat(xmatches);
    }
    return ys;
}