#include <tchem/linalg.hpp>

#include <Hderiva/diabatic.hpp>
#include <Hderiva/adiabatic.hpp>
#include <Hderiva/composite.hpp>

#include "common.hpp"

namespace train { namespace trust_region {

inline void reg_Jacobian(const size_t & thread, const std::shared_ptr<RegHam> & data,
at::Tensor & J, size_t & start) {
    // Get necessary diabatic quantities
    CL::utility::matrix<at::Tensor> xs = data->xs();
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = i; j < NStates; j++)
    xs[i][j].set_requires_grad(true);
    at::Tensor     Hd = Hdnets[thread]->forward(xs);
    at::Tensor   DqHd = Hderiva::DxHd(Hd, xs, data->JxqTs(), true);
    at::Tensor   DcHd = Hderiva::DcHd(Hd, Hdnets[thread]->elements->parameters());
    at::Tensor DcDqHd = Hderiva::DcDxHd(DqHd, Hdnets[thread]->elements->parameters());
    // Stop autograd tracking
      Hd.detach_();
    DqHd.detach_();
    // Get adiabatic representation
    at::Tensor energy, states;
    std::tie(energy, states) = define_adiabatz(Hd, DqHd,
        data->JqrT(), data->cartdim(), data->NStates(), data->dH());
    // Compute fitting parameter gradient in adiabatic prediction
    int64_t NStates_data = data->NStates();
    CL::utility::matrix<size_t> irreds = data->irreds();
    at::Tensor DcHa = tchem::linalg::UT_sy_U(DcHd, states);
    at::Tensor DqHa = tchem::linalg::UT_sy_U(DqHd, states);
    at::Tensor DcDqHa = Hderiva::DcDxHa(DqHa, DcHd, DcDqHd, energy, states);
    CL::utility::matrix<at::Tensor> DcSADqHa(NStates_data);
    for (size_t i = 0; i < NStates_data; i++)
    for (size_t j = i; j < NStates_data; j++)
    DcSADqHa[i][j] = data->cat(data->split2CNPI(DcDqHa[i][j]))[irreds[i][j]];
    // energy Jacobian
    at::Tensor J_E = unit * DcHa;
    for (size_t i = 0; i < NStates_data; i++) {
        J[start].copy_(data->sqrtweight_E(i) * J_E[i][i]);
        start++;
    }
    // (▽H)a Jacobian
    std::vector<at::Tensor> sqrtSs = data->sqrtSs();
    for (size_t i = 0; i < NStates_data; i++)
    for (size_t j = i; j < NStates_data; j++) {
        at::Tensor J_dH = data->sqrtweight_dH(i, j) * sqrtSs[irreds[i][j]].mm(DcSADqHa[i][j]);
        size_t stop = start + J_dH.size(0);
        J.slice(0, start, stop).copy_(J_dH);
        start = stop;
    }
}

inline void deg_Jacobian(const size_t & thread, const std::shared_ptr<DegHam> & data,
at::Tensor & J, size_t & start) {
    // Get necessary diabatic quantities
    CL::utility::matrix<at::Tensor> xs = data->xs();
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = i; j < NStates; j++)
    xs[i][j].set_requires_grad(true);
    at::Tensor     Hd = Hdnets[thread]->forward(xs);
    at::Tensor   DqHd = Hderiva::DxHd(Hd, xs, data->JxqTs(), true);
    at::Tensor   DcHd = Hderiva::DcHd(Hd, Hdnets[thread]->elements->parameters());
    at::Tensor DcDqHd = Hderiva::DcDxHd(DqHd, Hdnets[thread]->elements->parameters());
    // Stop autograd tracking
      Hd.detach_();
    DqHd.detach_();
    // Get composite representation
    at::Tensor eigval, eigvec;
    std::tie(eigval, eigvec) = define_composite(Hd, DqHd,
        data->JqrT(), data->cartdim(), data->H(), data->dH());
    // Compute fitting parameter gradient in composite prediction
    CL::utility::matrix<size_t> irreds = data->irreds();
    at::Tensor   Hc = tchem::linalg::UT_sy_U(  Hd, eigvec);
    at::Tensor DqHc = tchem::linalg::UT_sy_U(DqHd, eigvec);
    at::Tensor DcHc, DcDqHc;
    std::tie(DcHc, DcDqHc) = Hderiva::DcHc_DcDxHc(Hc, DqHc,
        DqHd, DcHd, DcDqHd, eigval, eigvec, data->S());
    CL::utility::matrix<at::Tensor> DcSADqHc(NStates);
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = i; j < NStates; j++)
    DcSADqHc[i][j] = data->cat(data->split2CNPI(DcDqHc[i][j]))[irreds[i][j]];
    // Hc Jacobian
    at::Tensor J_H = unit * DcHc;
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = i; j < NStates; j++)
    if (irreds[i][j] == 0) {
        J[start].copy_(data->sqrtweight_H(i, j) * J_H[i][j]);
        start++;
    }
    // (▽H)c Jacobian
    std::vector<at::Tensor> sqrtSs = data->sqrtSs();
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = i; j < NStates; j++) {
        at::Tensor J_dH = data->sqrtweight_dH(i, j) * sqrtSs[irreds[i][j]].mm(DcSADqHc[i][j]);
        size_t stop = start + J_dH.size(0);
        J.slice(0, start, stop).copy_(J_dH);
        start = stop;
    }
}

void Jacobian(double * JT, const double * c, const int32_t & M, const int32_t & N) {
    at::Tensor J = at::from_blob(JT, {N, M}, at::TensorOptions().dtype(torch::kFloat64));
    J.transpose_(0, 1);
    #pragma omp parallel for
    for (size_t thread = 0; thread < OMP_NUM_THREADS; thread++) {
        c2p(c, thread);
        size_t start = segstart[thread];
        for (const auto & data : regchunk[thread]) reg_Jacobian(thread, data, J, start);
        for (const auto & data : degchunk[thread]) deg_Jacobian(thread, data, J, start);
    }
}

void regularized_Jacobian(double * JT, const double * c, const int32_t & M, const int32_t & N) {
    at::Tensor J = at::from_blob(JT, {N, M}, at::TensorOptions().dtype(torch::kFloat64));
    J.transpose_(0, 1);
    #pragma omp parallel for
    for (size_t thread = 0; thread < OMP_NUM_THREADS; thread++) {
        c2p(c, thread);
        size_t start = segstart[thread];
        for (const auto & data : regchunk[thread]) reg_Jacobian(thread, data, J, start);
        for (const auto & data : degchunk[thread]) deg_Jacobian(thread, data, J, start);
    }
    at::Tensor regularization_block = J.slice(0, M - N, M);
    regularization_block.fill_(0.0);
    regularization_block.diagonal().copy_(regularization);
}

} // namespace trust_region
} // namespace train