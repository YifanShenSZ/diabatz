#include <tchem/linalg.hpp>

#include <Hderiva/diabatic.hpp>
#include <Hderiva/adiabatic.hpp>
#include <Hderiva/composite.hpp>

#include "common.hpp"

namespace train { namespace trust_region {

inline void reg_Jacobian(const size_t & thread, const std::shared_ptr<RegHam> & data,
at::Tensor & J, size_t & start) {
    // get necessary diabatic quantities
    CL::utility::matrix<at::Tensor> xs = data->xs();
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = i; j < NStates; j++)
    xs[i][j].set_requires_grad(true);
    at::Tensor     Hd = Hdnets[thread]->forward(xs);
    at::Tensor   DrHd = Hderiva::DxHd(Hd, xs, data->JxrTs(), true);
    at::Tensor   DcHd = Hderiva::DcHd(Hd, Hdnets[thread]->elements->parameters());
    at::Tensor DcDrHd = Hderiva::DcDxHd(DrHd, Hdnets[thread]->elements->parameters());
    // stop autograd tracking
      Hd.detach_();
    DrHd.detach_();
    // add the pretrained part
      Hd += data->pretrained_Hd  ();
    DrHd += data->pretrained_DrHd();
    // get adiabatic representation
    at::Tensor energy, states;
    std::tie(energy, states) = define_adiabatz(Hd, DrHd, data->NStates(), data->dH());
    // compute fitting parameter gradient in adiabatic prediction
    int64_t NStates_data = data->NStates();
    at::Tensor DcHa = tchem::linalg::UT_sy_U(DcHd, states);
    at::Tensor DrHa = tchem::linalg::UT_sy_U(DrHd, states);
    at::Tensor DcDrHa = Hderiva::DcDxHa(DrHa, DcHd, DcDrHd, energy, states);
    CL::utility::matrix<at::Tensor> DcSADQHa(NStates_data);
    for (size_t i = 0; i < NStates_data; i++)
    for (size_t j = i; j < NStates_data; j++)
    DcSADQHa[i][j] = data->C2Qs(data->irreds(i, j)).mm(DcDrHa[i][j]);
    // energy Jacobian
    at::Tensor J_E = unit * DcHa;
    for (size_t i = 0; i < NStates_data; i++) {
        J[start].copy_(data->sqrtweight_E(i) * J_E[i][i]);
        start++;
    }
    // (▽H)a Jacobian
    for (size_t i = 0; i < NStates_data; i++)
    for (size_t j = i; j < NStates_data; j++) {
        at::Tensor J_dH = data->sqrtweight_dH(i, j) * data->sqrtSQs(data->irreds(i, j)).mm(DcSADQHa[i][j]);
        size_t stop = start + J_dH.size(0);
        J.slice(0, start, stop).copy_(J_dH);
        start = stop;
    }
}

inline void deg_Jacobian(const size_t & thread, const std::shared_ptr<DegHam> & data,
at::Tensor & J, size_t & start) {
    // get necessary diabatic quantities
    CL::utility::matrix<at::Tensor> xs = data->xs();
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = i; j < NStates; j++)
    xs[i][j].set_requires_grad(true);
    at::Tensor     Hd = Hdnets[thread]->forward(xs);
    at::Tensor   DrHd = Hderiva::DxHd(Hd, xs, data->JxrTs(), true);
    at::Tensor   DcHd = Hderiva::DcHd(Hd, Hdnets[thread]->elements->parameters());
    at::Tensor DcDrHd = Hderiva::DcDxHd(DrHd, Hdnets[thread]->elements->parameters());
    // stop autograd tracking
      Hd.detach_();
    DrHd.detach_();
    // add the pretrained part
      Hd += data->pretrained_Hd  ();
    DrHd += data->pretrained_DrHd();
    // get composite representation
    at::Tensor eigval, eigvec;
    std::tie(eigval, eigvec) = define_composite(Hd, DrHd, data->H(), data->dH());
    // compute fitting parameter gradient in composite prediction
    at::Tensor   Hc = tchem::linalg::UT_sy_U(  Hd, eigvec);
    at::Tensor DrHc = tchem::linalg::UT_sy_U(DrHd, eigvec);
    at::Tensor DcHc, DcDrHc;
    std::tie(DcHc, DcDrHc) = Hderiva::DcHc_DcDxHc(Hc, DrHc,
        DrHd, DcHd, DcDrHd, eigval, eigvec);
    CL::utility::matrix<at::Tensor> DcSADQHc(NStates);
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = i; j < NStates; j++)
    DcSADQHc[i][j] = data->C2Qs(data->irreds(i, j)).mm(DcDrHc[i][j]);
    // Hc Jacobian
    at::Tensor J_H = unit * DcHc;
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = i; j < NStates; j++)
    if (data->irreds(i, j) == 0) {
        J[start].copy_(data->sqrtweight_H(i, j) * J_H[i][j]);
        start++;
    }
    // (▽H)c Jacobian
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = i; j < NStates; j++) {
        at::Tensor J_dH = data->sqrtweight_dH(i, j) * data->sqrtSQs(data->irreds(i, j)).mm(DcSADQHc[i][j]);
        size_t stop = start + J_dH.size(0);
        J.slice(0, start, stop).copy_(J_dH);
        start = stop;
    }
}

inline void energy_Jacobian(const size_t & thread, const std::shared_ptr<Energy> & data,
at::Tensor & J, size_t & start) {
    // get energy and its gradient over fitting parameters
    CL::utility::matrix<at::Tensor> xs = data->xs();
    at::Tensor   Hd = Hdnets[thread]->forward(xs);
    at::Tensor DcHd = Hderiva::DcHd(Hd, Hdnets[thread]->elements->parameters());
    Hd.detach_();
    Hd += data->pretrained_Hd(); // add the pretrained part
    at::Tensor energy, states;
    std::tie(energy, states) = Hd.symeig(true);
    at::Tensor DcHa = tchem::linalg::UT_sy_U(DcHd, states);
    // energy Jacobian
    at::Tensor J_E = unit * DcHa;
    for (size_t i = 0; i < data->NStates(); i++) {
        J[start].copy_(data->sqrtweight_E(i) * J_E[i][i]);
        start++;
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
        for (const auto & data : energy_chunk[thread]) energy_Jacobian(thread, data, J, start);
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
        for (const auto & data : energy_chunk[thread]) energy_Jacobian(thread, data, J, start);
    }
    at::Tensor regularization_block = J.slice(0, M - N, M);
    regularization_block.fill_(0.0);
    regularization_block.diagonal().copy_(regularization);
}

} // namespace trust_region
} // namespace train