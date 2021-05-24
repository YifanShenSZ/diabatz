#include <tchem/linalg.hpp>

#include <Hderiva/diabatic.hpp>

#include "common.hpp"

namespace train { namespace trust_region {

inline void reg_residue(const size_t & thread, const std::shared_ptr<RegHam> & data,
double * r, size_t & start) {
    // Get necessary diabatic quantities
    CL::utility::matrix<at::Tensor> xs = data->xs();
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = i; j < NStates; j++)
    xs[i][j].set_requires_grad(true);
    at::Tensor   Hd = Hdnets[thread]->forward(xs);
    at::Tensor DqHd = Hderiva::DxHd(Hd, xs, data->JxqTs());
    // Stop autograd tracking
    Hd.detach_();
    // Get adiabatic representation
    at::Tensor energy, states;
    std::tie(energy, states) = define_adiabatz(Hd, DqHd,
        data->JqrT(), data->cartdim(), data->NStates(), data->dH());
    // Make prediction in adiabatic representation
    int64_t NStates_data = data->NStates();
    const CL::utility::matrix<size_t> & irreds = data->irreds();
    energy = energy.slice(0, 0, NStates_data);
    at::Tensor DqHa = tchem::linalg::UT_sy_U(DqHd, states);
    CL::utility::matrix<at::Tensor> SADqHa(NStates_data);
    for (size_t i = 0; i < NStates_data; i++)
    for (size_t j = i; j < NStates_data; j++)
    SADqHa[i][j] = data->cat(data->split2CNPI(DqHa[i][j]))[irreds[i][j]];
    // energy residue
    at::Tensor r_E = unit * (energy - data->energy());
    for (size_t i = 0; i < NStates_data; i++) {
        r[start] = data->sqrtweight_E(i) * r_E[i].item<double>();
        start++;
    }
    // (▽H)a residue
    const std::vector<at::Tensor> & sqrtSs = data->sqrtSs();
    const CL::utility::matrix<at::Tensor> & SAdH = data->SAdH();
    for (size_t i = 0; i < NStates_data; i++)
    for (size_t j = i; j < NStates_data; j++) {
        at::Tensor r_dH = data->sqrtweight_dH(i, j) * sqrtSs[irreds[i][j]].mv(SADqHa[i][j] - SAdH[i][j]);
        std::memcpy(&(r[start]), r_dH.data_ptr<double>(), r_dH.numel() * sizeof(double));
        start += r_dH.numel();
    }
}

inline void deg_residue(const size_t & thread, const std::shared_ptr<DegHam> & data,
double * r, size_t & start) {
    // Get necessary diabatic quantities
    CL::utility::matrix<at::Tensor> xs = data->xs();
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = i; j < NStates; j++)
    xs[i][j].set_requires_grad(true);
    at::Tensor   Hd = Hdnets[thread]->forward(xs);
    at::Tensor DqHd = Hderiva::DxHd(Hd, xs, data->JxqTs());
    // Stop autograd tracking
    Hd.detach_();
    // Get composite representation
    at::Tensor eigval, eigvec;
    std::tie(eigval, eigvec) = define_composite(Hd, DqHd,
        data->JqrT(), data->cartdim(), data->H(), data->dH());
    // Make prediction in composite representation
    const CL::utility::matrix<size_t> & irreds = data->irreds();
    at::Tensor   Hc = tchem::linalg::UT_sy_U(  Hd, eigvec);
    at::Tensor DqHc = tchem::linalg::UT_sy_U(DqHd, eigvec);
    CL::utility::matrix<at::Tensor> SADqHc(NStates);
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = i; j < NStates; j++)
    SADqHc[i][j] = data->cat(data->split2CNPI(DqHc[i][j]))[irreds[i][j]];
    // Hc residue
    at::Tensor r_H = unit * (Hc - data->H());
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = i; j < NStates; j++)
    if (irreds[i][j] == 0) {
        r[start] = data->sqrtweight_H(i, j) * r_H[i][j].item<double>();
        start++;
    }
    // (▽H)c residue
    const std::vector<at::Tensor> & sqrtSs = data->sqrtSs();
    const CL::utility::matrix<at::Tensor> & SAdH = data->SAdH();
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = i; j < NStates; j++) {
        at::Tensor r_dH = data->sqrtweight_dH(i, j) * sqrtSs[irreds[i][j]].mv(SADqHc[i][j] - SAdH[i][j]);
        std::memcpy(&(r[start]), r_dH.data_ptr<double>(), r_dH.numel() * sizeof(double));
        start += r_dH.numel();
    }
}

void residue(double * r, const double * c, const int32_t & M, const int32_t & N) {
    #pragma omp parallel for
    for (size_t thread = 0; thread < OMP_NUM_THREADS; thread++) {
        c2p(c, thread);
        size_t start = segstart[thread];
        for (const auto & data : regchunk[thread]) reg_residue(thread, data, r, start);
        for (const auto & data : degchunk[thread]) deg_residue(thread, data, r, start);
    }
}

void regularized_residue(double * r, const double * c, const int32_t & M, const int32_t & N) {
    #pragma omp parallel for
    for (size_t thread = 0; thread < OMP_NUM_THREADS; thread++) {
        c2p(c, thread);
        size_t start = segstart[thread];
        for (const auto & data : regchunk[thread]) reg_residue(thread, data, r, start);
        for (const auto & data : degchunk[thread]) deg_residue(thread, data, r, start);
    }
    c10::TensorOptions top = c10::TensorOptions().dtype(torch::kFloat64);
    at::Tensor residue = at::from_blob(r, M, top),
               p       = at::from_blob(const_cast<double *>(c), N, top);
    residue.slice(0, M - N, M).copy_(regularization * (p - prior));
}

} // namespace trust_region
} // namespace train