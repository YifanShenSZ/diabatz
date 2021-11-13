#include "global.hpp"

double threshold = 1e-6;

std::shared_ptr<SASIC::SASICSet> sasicset;

// Given Cartesian coordinate r,
// return symmetry adapted internal coordinates
// and their transposed Jacobian over r
std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> cart2int(const at::Tensor & r) {
    assert(("Define symmetry adaptated and scaled internal coordinate before use", sasicset));
    // Cartesian coordinate -> internal coordinate
    at::Tensor q, JT;
    std::tie(q, JT) = sasicset->compute_IC_J(r);
    JT.transpose_(0, 1);
    q.set_requires_grad(true);
    // Internal coordinate -> symmetry adapted internal coordinate
    std::vector<at::Tensor> qs = (*sasicset)(q);
    std::vector<at::Tensor> JTs = std::vector<at::Tensor>(qs.size());
    for (size_t i = 0; i < qs.size(); i++) {
        JTs[i] = qs[i].new_empty({qs[i].size(0), q.size(0)});
        for (size_t j = 0; j < qs[i].size(0); j++) {
            std::vector<at::Tensor> g = torch::autograd::grad({qs[i][j]}, {q}, {}, true);
            JTs[i][j].copy_(g[0]);
        }
        JTs[i].transpose_(0, 1);
        JTs[i] = JT.mm(JTs[i]);
    }
    return std::make_tuple(qs, JTs);
}