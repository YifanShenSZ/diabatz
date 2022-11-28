#include "../include/global.hpp"

std::shared_ptr<SASDIC::SASDICSet> sasicset;

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
    // free autograd graph
    for (at::Tensor & q : qs) q.detach_();
    return std::make_tuple(qs, Js);
}

std::shared_ptr<obnet::symat> Hdnet1, Hdnet2;

std::shared_ptr<InputGenerator> input_generator1, input_generator2;

std::tuple<CL::utility::matrix<at::Tensor>, CL::utility::matrix<at::Tensor>>
int2input1(const std::vector<at::Tensor> & qs) {
    size_t NStates = Hdnet1->NStates();
    CL::utility::matrix<at::Tensor> xs(NStates), JTs(NStates);
    std::tie(xs, JTs) = input_generator1->compute_x_JT(qs);
    return std::make_tuple(xs, JTs);
}

std::tuple<CL::utility::matrix<at::Tensor>, CL::utility::matrix<at::Tensor>>
int2input2(const std::vector<at::Tensor> & qs) {
    size_t NStates = Hdnet2->NStates();
    CL::utility::matrix<at::Tensor> xs(NStates), JTs(NStates);
    std::tie(xs, JTs) = input_generator2->compute_x_JT(qs);
    return std::make_tuple(xs, JTs);
}

at::Tensor regularization, prior;

// the "unit" of energy, accounting for the unit difference between energy and gradient
double unit, unit_square;