#include <Foptim/line-search_2nd/BFGS.hpp>

#include "../../include/global.hpp"
#include "../../include/Hd_extension.hpp"

namespace {

at::Tensor init_guess_;
int64_t target_state_;

void energy(double & energy, const double * free_intgeom, const int32_t & free_intdim) {
    at::Tensor q_free = at::from_blob(const_cast<double *>(free_intgeom), free_intdim,
                                      at::TensorOptions().dtype(torch::kFloat64));
    at::Tensor q = fixed_intcoord->vector_free2total(q_free);
    at::Tensor r = int2cart(q, init_guess_, intcoordset);
    at::Tensor e = compute_energy(r);
    energy = e[target_state_].item<double>();
}

void energy_grad(double & energy, double * grad, const double * free_intgeom, const int32_t & free_intdim) {
    at::Tensor q_free = at::from_blob(const_cast<double *>(free_intgeom), free_intdim,
                                      at::TensorOptions().dtype(torch::kFloat64));
    at::Tensor q = fixed_intcoord->vector_free2total(q_free);
    at::Tensor r = int2cart(q, init_guess_, intcoordset);
    at::Tensor e, dHa;
    std::tie(e, dHa) = compute_energy_dHa(r);
    at::Tensor intgrad = intcoordset->gradient_cart2int(r, dHa[target_state_][target_state_]);
    at::Tensor free_intgrad = fixed_intcoord->vector_total2free(intgrad);
    energy = e[target_state_].item<double>();
    std::memcpy(grad, free_intgrad.data_ptr<double>(), free_intdim * sizeof(double));
}

void Hessian(double * Hessian, const double * free_intgeom, const int32_t & free_intdim) {
    at::Tensor q_free = at::from_blob(const_cast<double *>(free_intgeom), free_intdim,
                                      at::TensorOptions().dtype(torch::kFloat64));
    at::Tensor q = fixed_intcoord->vector_free2total(q_free);
    at::Tensor r = int2cart(q, init_guess_, intcoordset);
    at::Tensor e, dHa;
    std::tie(e, dHa) = compute_energy_dHa(r);
    at::Tensor cartHess = compute_ddHa(r)[target_state_][target_state_];
    at::Tensor intHess = intcoordset->Hessian_cart2int(r, dHa[target_state_][target_state_], cartHess);
    at::Tensor free_intHess = fixed_intcoord->matrix_total2free(intHess);
    std::memcpy(Hessian, free_intHess.data_ptr<double>(), free_intdim * free_intdim * sizeof(double));
}

}

at::Tensor search_minimum_adiabatic(const at::Tensor& _init_guess, const int64_t& _target_state) {
    init_guess_ = _init_guess;
    target_state_ = _target_state;
    at::Tensor q = (*intcoordset)(_init_guess);
    at::Tensor q_free = fixed_intcoord->vector_total2free(q);
    Foptim::BFGS(energy, energy_grad, Hessian,
                 q_free.data_ptr<double>(), q_free.size(0),
                 20, 100, 1e-4, 1e-4);
    q = fixed_intcoord->vector_free2total(q_free);
    at::Tensor r = int2cart(q, _init_guess, intcoordset);
    return r;
}