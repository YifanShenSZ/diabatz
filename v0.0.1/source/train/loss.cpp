#include <tchem/linalg.hpp>

#include <Hderiva/diabatic.hpp>

#include "common.hpp"

namespace train { namespace line_search {

inline void reg_loss(const size_t & thread, const std::shared_ptr<RegHam> & data,
double & loss) {
    // get necessary diabatic quantities
    CL::utility::matrix<at::Tensor> xs = data->xs();
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = i; j < NStates; j++)
    xs[i][j].set_requires_grad(true);
    at::Tensor   Hd = Hdnets[thread]->forward(xs);
    at::Tensor DqHd = Hderiva::DxHd(Hd, xs, data->JxqTs());
    // stop autograd tracking
    Hd.detach_();
    // get adiabatic representation
    at::Tensor energy, states;
    std::tie(energy, states) = define_adiabatz(Hd, DqHd,
        data->JqrT(), data->cartdim(), data->NStates(), data->dH());
    // make prediction in adiabatic representation
    int64_t NStates_data = data->NStates();
    energy = energy.slice(0, 0, NStates_data);
    at::Tensor DqHa = tchem::linalg::UT_sy_U(DqHd, states);
    CL::utility::matrix<at::Tensor> SADQHa(NStates_data);
    for (size_t i = 0; i < NStates_data; i++)
    for (size_t j = i; j < NStates_data; j++)
    SADQHa[i][j] = data->C2Qs(data->irreds(i, j)).mv(data->JqrT().mv(DqHa[i][j]));
    // energy loss
    at::Tensor r_E = unit * (energy - data->energy());
    for (size_t i = 0; i < NStates_data; i++) {
        double this_state = r_E[i].item<double>();
        loss += data->weight_E(i) * this_state * this_state;
    }
    // (▽H)a loss
    for (size_t i = 0; i < NStates_data; i++)
    for (size_t j = i; j < NStates_data; j++) {
        at::Tensor r_dH = data->sqrtSQs(data->irreds(i, j)).mv(SADQHa[i][j] - data->SAdH(i, j));
        loss += data->weight_dH(i, j) * (r_dH.dot(r_dH)).item<double>();
    }
}

inline void deg_loss(const size_t & thread, const std::shared_ptr<DegHam> & data,
double & loss) {
    // get necessary diabatic quantities
    CL::utility::matrix<at::Tensor> xs = data->xs();
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = i; j < NStates; j++)
    xs[i][j].set_requires_grad(true);
    at::Tensor   Hd = Hdnets[thread]->forward(xs);
    at::Tensor DqHd = Hderiva::DxHd(Hd, xs, data->JxqTs());
    // stop autograd tracking
    Hd.detach_();
    // get composite representation
    at::Tensor eigval, eigvec;
    std::tie(eigval, eigvec) = define_composite(Hd, DqHd,
        data->JqrT(), data->cartdim(), data->H(), data->dH());
    // make prediction in composite representation
    at::Tensor   Hc = tchem::linalg::UT_sy_U(  Hd, eigvec);
    at::Tensor DqHc = tchem::linalg::UT_sy_U(DqHd, eigvec);
    CL::utility::matrix<at::Tensor> SADQHc(NStates);
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = i; j < NStates; j++)
    SADQHc[i][j] = data->C2Qs(data->irreds(i, j)).mv(data->JqrT().mv(DqHc[i][j]));
    // Hc loss
    at::Tensor r_H = unit * (Hc - data->H());
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = i; j < NStates; j++)
    if (data->irreds(i, j) == 0) {
        double this_state = r_H[i][j].item<double>();
        loss += data->weight_H(i, j) * this_state * this_state;
    }
    // (▽H)c loss
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = i; j < NStates; j++) {
        at::Tensor r_dH = data->sqrtSQs(data->irreds(i, j)).mv(SADQHc[i][j] - data->SAdH(i, j));
        loss += data->weight_dH(i, j) * (r_dH.dot(r_dH)).item<double>();
    }
}

inline void energy_loss(const size_t & thread, const std::shared_ptr<Energy> & data,
double & loss) {
    // get energy
    CL::utility::matrix<at::Tensor> xs = data->xs();
    at::Tensor Hd = Hdnets[thread]->forward(xs);
    Hd.detach_();
    at::Tensor energy, states;
    std::tie(energy, states) = Hd.symeig();
    // energy loss
    int64_t NStates_data = data->NStates();
    energy = energy.slice(0, 0, NStates_data);
    at::Tensor r_E = unit * (energy - data->energy());
    for (size_t i = 0; i < NStates_data; i++) {
        double this_state = r_E[i].item<double>();
        loss += data->weight_E(i) * this_state * this_state;
    }
}

void loss(double & l, const double * c, const int32_t & N) {
    #pragma omp parallel for
    for (size_t thread = 0; thread < OMP_NUM_THREADS; thread++) {
        double & this_loss = losses[thread];
        this_loss = 0.0;
        c2p(c, thread);
        for (const auto & data : regchunk[thread]) reg_loss(thread, data, this_loss);
        for (const auto & data : degchunk[thread]) deg_loss(thread, data, this_loss);
        for (const auto & data : energy_chunk[thread]) energy_loss(thread, data, this_loss);
    }
    l = 0.5 * std::accumulate(losses.begin(), losses.end(), 0.0);
}

void regularized_loss(double & l, const double * c, const int32_t & N) {
    #pragma omp parallel for
    for (size_t thread = 0; thread < OMP_NUM_THREADS; thread++) {
        double & this_loss = losses[thread];
        this_loss = 0.0;
        c2p(c, thread);
        for (const auto & data : regchunk[thread]) reg_loss(thread, data, this_loss);
        for (const auto & data : degchunk[thread]) deg_loss(thread, data, this_loss);
        for (const auto & data : energy_chunk[thread]) energy_loss(thread, data, this_loss);
    }
    l = std::accumulate(losses.begin(), losses.end(), 0.0);
    // regularization
    at::Tensor p = at::from_blob(const_cast<double *>(c), N, c10::TensorOptions().dtype(torch::kFloat64));
    at::Tensor r = regularization * (p - prior);
    l += (r.dot(r)).item<double>();
    l *= 0.5;
}

} // namespace line_search
} // namespace train