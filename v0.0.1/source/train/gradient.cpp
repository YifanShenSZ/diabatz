#include <tchem/linalg.hpp>

#include <Hderiva/diabatic.hpp>
#include <Hderiva/adiabatic.hpp>
#include <Hderiva/composite.hpp>

#include "common.hpp"

namespace train { namespace line_search {

inline void reg_loss_gradient(const size_t & thread, const std::shared_ptr<RegHam> & data,
double & loss, at::Tensor & gradient) {
    // get necessary diabatic quantities
    CL::utility::matrix<at::Tensor> xs = data->xs();
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = i; j < NStates; j++)
    xs[i][j].set_requires_grad(true);
    at::Tensor     Hd = Hdnets[thread]->forward(xs);
    at::Tensor   DqHd = Hderiva::DxHd(Hd, xs, data->JxqTs(), true);
    at::Tensor   DcHd = Hderiva::DcHd(Hd, Hdnets[thread]->elements->parameters());
    at::Tensor DcDqHd = Hderiva::DcDxHd(DqHd, Hdnets[thread]->elements->parameters());
    // stop autograd tracking
      Hd.detach_();
    DqHd.detach_();
    // get adiabatic representation
    at::Tensor energy, states;
    std::tie(energy, states) = define_adiabatz(Hd, DqHd,
        data->JqrT(), data->cartdim(), data->NStates(), data->dH());
    // compute fitting parameter gradient in adiabatic prediction
    int64_t NStates_data = data->NStates();
    at::Tensor DcHa = tchem::linalg::UT_sy_U(DcHd, states);
    at::Tensor DqHa = tchem::linalg::UT_sy_U(DqHd, states);
    at::Tensor DcDqHa = Hderiva::DcDxHa(DqHa, DcHd, DcDqHd, energy, states);
    CL::utility::matrix<at::Tensor> SADQHa(NStates_data), DcSADQHa(NStates_data);
    for (size_t i = 0; i < NStates_data; i++)
    for (size_t j = i; j < NStates_data; j++) {
          SADQHa[i][j] = data->C2Qs(data->irreds(i, j)).mv(data->JqrT().mv(  DqHa[i][j]));
        DcSADQHa[i][j] = data->C2Qs(data->irreds(i, j)).mm(data->JqrT().mm(DcDqHa[i][j]));
    }
    // energy loss and gradient
    at::Tensor r_E = unit * (energy - data->energy()),
               J_E = unit * DcHa;
    for (size_t i = 0; i < NStates_data; i++) {
        double this_state = r_E[i].item<double>();
        loss += data->weight_E(i) * this_state * this_state;
        gradient += data->weight_E(i) * r_E[i] * J_E[i][i];
    }
    // (▽H)a loss and gradient
    for (size_t i = 0; i < NStates_data; i++)
    for (size_t j = i; j < NStates_data; j++) {
        at::Tensor r_dH = data->sqrtSQs(data->irreds(i, j)).mv(SADQHa[i][j] - data->SAdH(i, j)),
                   J_dH = data->sqrtSQs(data->irreds(i, j)).mm(DcSADQHa[i][j]);
        loss += data->weight_dH(i, j) * (r_dH.dot(r_dH)).item<double>();
        gradient += data->weight_dH(i, j) * at::matmul(r_dH, J_dH);
    }
}

inline void deg_loss_gradient(const size_t & thread, const std::shared_ptr<DegHam> & data,
double & loss, at::Tensor & gradient) {
    // get necessary diabatic quantities
    CL::utility::matrix<at::Tensor> xs = data->xs();
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = i; j < NStates; j++)
    xs[i][j].set_requires_grad(true);
    at::Tensor     Hd = Hdnets[thread]->forward(xs);
    at::Tensor   DqHd = Hderiva::DxHd(Hd, xs, data->JxqTs(), true);
    at::Tensor   DcHd = Hderiva::DcHd(Hd, Hdnets[thread]->elements->parameters());
    at::Tensor DcDqHd = Hderiva::DcDxHd(DqHd, Hdnets[thread]->elements->parameters());
    // stop autograd tracking
      Hd.detach_();
    DqHd.detach_();
    // get composite representation
    at::Tensor eigval, eigvec;
    std::tie(eigval, eigvec) = define_composite(Hd, DqHd,
        data->JqrT(), data->cartdim(), data->H(), data->dH());
    // compute fitting parameter gradient in composite prediction
    at::Tensor   Hc = tchem::linalg::UT_sy_U(  Hd, eigvec);
    at::Tensor DqHc = tchem::linalg::UT_sy_U(DqHd, eigvec);
    at::Tensor DcHc, DcDqHc;
    std::tie(DcHc, DcDqHc) = Hderiva::DcHc_DcDxHc(Hc, DqHc,
        DqHd, DcHd, DcDqHd, eigval, eigvec, data->Sq());
    CL::utility::matrix<at::Tensor> SADQHc(NStates), DcSADQHc(NStates);
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = i; j < NStates; j++) {
          SADQHc[i][j] = data->C2Qs(data->irreds(i, j)).mv(data->JqrT().mv(  DqHc[i][j]));
        DcSADQHc[i][j] = data->C2Qs(data->irreds(i, j)).mm(data->JqrT().mm(DcDqHc[i][j]));
    }
    // Hc loss and gradient
    at::Tensor r_H = unit * (Hc - data->H()),
               J_H = unit * DcHc;
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = i; j < NStates; j++)
    if (data->irreds(i, j) == 0) {
        double this_state = r_H[i][j].item<double>();
        loss += data->weight_H(i, j) * this_state * this_state;
        gradient += data->weight_H(i, j) * r_H[i][j] * J_H[i][j];
    }
    // (▽H)c loss and gradient
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = i; j < NStates; j++) {
        at::Tensor r_dH = data->sqrtSQs(data->irreds(i, j)).mv(SADQHc[i][j] - data->SAdH(i, j)),
                   J_dH = data->sqrtSQs(data->irreds(i, j)).mm(DcSADQHc[i][j]);
        loss += data->weight_dH(i, j) * (r_dH.dot(r_dH)).item<double>();
        gradient += data->weight_dH(i, j) * at::matmul(r_dH, J_dH);
    }
}

inline void energy_loss_gradient(const size_t & thread, const std::shared_ptr<Energy> & data,
double & loss, at::Tensor & gradient) {
    // get energy and its gradient over fitting parameters
    CL::utility::matrix<at::Tensor> xs = data->xs();
    at::Tensor   Hd = Hdnets[thread]->forward(xs);
    at::Tensor DcHd = Hderiva::DcHd(Hd, Hdnets[thread]->elements->parameters());
    Hd.detach_();
    at::Tensor energy, states;
    std::tie(energy, states) = Hd.symeig(true);
    at::Tensor DcHa = tchem::linalg::UT_sy_U(DcHd, states);
    // energy gradient
    int64_t NStates_data = data->NStates();
    energy = energy.slice(0, 0, NStates_data);
    at::Tensor r_E = unit * (energy - data->energy()),
               J_E = unit * DcHa;
    for (size_t i = 0; i < NStates_data; i++) {
        double this_state = r_E[i].item<double>();
        loss += data->weight_E(i) * this_state * this_state;
        gradient += data->weight_E(i) * r_E[i] * J_E[i][i];
    }
}

void loss_gradient(double & l, double * g, const double * c, const int32_t & N) {
    #pragma omp parallel for
    for (size_t thread = 0; thread < OMP_NUM_THREADS; thread++) {
        double & this_loss = losses[thread];
        this_loss = 0.0;
        at::Tensor & this_gradient = gradients[thread];
        this_gradient.fill_(0.0);
        c2p(c, thread);
        for (const auto & data : regchunk[thread]) reg_loss_gradient(thread, data, this_loss, this_gradient);
        for (const auto & data : degchunk[thread]) deg_loss_gradient(thread, data, this_loss, this_gradient);
        for (const auto & data : energy_chunk[thread]) energy_loss_gradient(thread, data, this_loss, this_gradient);
    }
    l = 0.5 * std::accumulate(losses.begin(), losses.end(), 0.0);
    at::Tensor gradient = at::from_blob(g, N, at::TensorOptions().dtype(torch::kFloat64));
    gradient.fill_(0.0);
    for (size_t thread = 0; thread < OMP_NUM_THREADS; thread++) gradient += gradients[thread];
}

void regularized_loss_gradient(double & l, double * g, const double * c, const int32_t & N) {
    #pragma omp parallel for
    for (size_t thread = 0; thread < OMP_NUM_THREADS; thread++) {
        double & this_loss = losses[thread];
        this_loss = 0.0;
        at::Tensor & this_gradient = gradients[thread];
        this_gradient.fill_(0.0);
        c2p(c, thread);
        for (const auto & data : regchunk[thread]) reg_loss_gradient(thread, data, this_loss, this_gradient);
        for (const auto & data : degchunk[thread]) deg_loss_gradient(thread, data, this_loss, this_gradient);
        for (const auto & data : energy_chunk[thread]) energy_loss_gradient(thread, data, this_loss, this_gradient);
    }
    l = std::accumulate(losses.begin(), losses.end(), 0.0);
    at::Tensor gradient = at::from_blob(g, N, at::TensorOptions().dtype(torch::kFloat64));
    gradient.fill_(0.0);
    for (size_t thread = 0; thread < OMP_NUM_THREADS; thread++) gradient += gradients[thread];
    // regularization
    at::Tensor p = at::from_blob(const_cast<double *>(c), N, c10::TensorOptions().dtype(torch::kFloat64));
    at::Tensor r = regularization * (p - prior);
    l += (r.dot(r)).item<double>();
    l *= 0.5;
    gradient += regularization * r;
}

} // namespace line_search
} // namespace train