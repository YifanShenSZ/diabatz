#include <tchem/linalg.hpp>

#include <Hderiva/diabatic.hpp>

#include "common.hpp"

namespace train { namespace trust_region {

inline void reg_residue(const size_t & thread, const std::shared_ptr<RegHam> & data,
double * r, size_t & start) {
    // get necessary diabatic quantities
    CL::utility::matrix<at::Tensor> xs = data->xs();
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = i; j < NStates; j++)
    xs[i][j].set_requires_grad(true);
    at::Tensor   Hd = Hdnets[thread]->forward(xs);
    at::Tensor DrHd = Hderiva::DxHd(Hd, xs, data->JxrTs());
    // stop autograd tracking
    Hd.detach_();
    // get adiabatic representation
    at::Tensor energy, states;
    std::tie(energy, states) = define_adiabatz(Hd, DrHd, data->NStates(), data->dH());
    // make prediction in adiabatic representation
    int64_t NStates_data = data->NStates();
    energy = energy.slice(0, 0, NStates_data);
    at::Tensor DrHa = tchem::linalg::UT_sy_U(DrHd, states);
    CL::utility::matrix<at::Tensor> SADQHa(NStates_data);
    for (size_t i = 0; i < NStates_data; i++)
    for (size_t j = i; j < NStates_data; j++)
    SADQHa[i][j] = data->C2Qs(data->irreds(i, j)).mv(DrHa[i][j]);
    // energy residue
    at::Tensor r_E = unit * (energy - data->energy());
    for (size_t i = 0; i < NStates_data; i++) {
        r[start] = data->sqrtweight_E(i) * r_E[i].item<double>();
        start++;
    }
    // (▽H)a residue
    for (size_t i = 0; i < NStates_data; i++)
    for (size_t j = i; j < NStates_data; j++) {
        at::Tensor r_dH = data->sqrtweight_dH(i, j) * data->sqrtSQs(data->irreds(i, j)).mv(SADQHa[i][j] - data->SAdH(i, j));
        std::memcpy(&(r[start]), r_dH.data_ptr<double>(), r_dH.numel() * sizeof(double));
        start += r_dH.numel();
    }
}

inline void deg_residue(const size_t & thread, const std::shared_ptr<DegHam> & data,
double * r, size_t & start) {
    // get necessary diabatic quantities
    CL::utility::matrix<at::Tensor> xs = data->xs();
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = i; j < NStates; j++)
    xs[i][j].set_requires_grad(true);
    at::Tensor   Hd = Hdnets[thread]->forward(xs);
    at::Tensor DrHd = Hderiva::DxHd(Hd, xs, data->JxrTs());
    // stop autograd tracking
    Hd.detach_();
    // get composite representation
    at::Tensor eigval, eigvec;
    std::tie(eigval, eigvec) = define_composite(Hd, DrHd, data->H(), data->dH());
    // make prediction in composite representation
    at::Tensor   Hc = tchem::linalg::UT_sy_U(  Hd, eigvec);
    at::Tensor DrHc = tchem::linalg::UT_sy_U(DrHd, eigvec);
    CL::utility::matrix<at::Tensor> SADQHc(NStates);
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = i; j < NStates; j++)
    SADQHc[i][j] = data->C2Qs(data->irreds(i, j)).mv(DrHc[i][j]);
    // Hc residue
    at::Tensor r_H = unit * (Hc - data->H());
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = i; j < NStates; j++)
    if (data->irreds(i, j) == 0) {
        r[start] = data->sqrtweight_H(i, j) * r_H[i][j].item<double>();
        start++;
    }
    // (▽H)c residue
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = i; j < NStates; j++) {
        at::Tensor r_dH = data->sqrtweight_dH(i, j) * data->sqrtSQs(data->irreds(i, j)).mv(SADQHc[i][j] - data->SAdH(i, j));
        std::memcpy(&(r[start]), r_dH.data_ptr<double>(), r_dH.numel() * sizeof(double));
        start += r_dH.numel();
    }
}

inline void energy_residue(const size_t & thread, const std::shared_ptr<Energy> & data,
double * r, size_t & start) {
    // get energy
    CL::utility::matrix<at::Tensor> xs = data->xs();
    at::Tensor Hd = Hdnets[thread]->forward(xs);
    Hd.detach_();
    at::Tensor energy, states;
    std::tie(energy, states) = Hd.symeig();
    // energy residue
    int64_t NStates_data = data->NStates();
    energy = energy.slice(0, 0, NStates_data);
    at::Tensor r_E = unit * (energy - data->energy());
    for (size_t i = 0; i < NStates_data; i++) {
        r[start] = data->sqrtweight_E(i) * r_E[i].item<double>();
        start++;
    }
}

void residue(double * r, const double * c, const int32_t & M, const int32_t & N) {
    #pragma omp parallel for
    for (size_t thread = 0; thread < OMP_NUM_THREADS; thread++) {
        c2p(c, thread);
        size_t start = segstart[thread];
        for (const auto & data : regchunk[thread]) reg_residue(thread, data, r, start);
        for (const auto & data : degchunk[thread]) deg_residue(thread, data, r, start);
        for (const auto & data : energy_chunk[thread]) energy_residue(thread, data, r, start);
    }
}

void regularized_residue(double * r, const double * c, const int32_t & M, const int32_t & N) {
    #pragma omp parallel for
    for (size_t thread = 0; thread < OMP_NUM_THREADS; thread++) {
        c2p(c, thread);
        size_t start = segstart[thread];
        for (const auto & data : regchunk[thread]) reg_residue(thread, data, r, start);
        for (const auto & data : degchunk[thread]) deg_residue(thread, data, r, start);
        for (const auto & data : energy_chunk[thread]) energy_residue(thread, data, r, start);
    }
    c10::TensorOptions top = c10::TensorOptions().dtype(torch::kFloat64);
    at::Tensor residue = at::from_blob(r, M, top),
               p       = at::from_blob(const_cast<double *>(c), N, top);
    residue.slice(0, M - N, M).copy_(regularization * (p - prior));
}

} // namespace trust_region
} // namespace train