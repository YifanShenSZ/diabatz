#include <Foptim/BFGS.hpp>

#include "../include/global.hpp"
#include "../include/Hd_extension.hpp"

namespace {

at::Tensor init_guess_;

void energy(double & energy, const double * free_intgeom, const int32_t & free_intdim) {
    at::Tensor q_free = at::from_blob(const_cast<double *>(free_intgeom), free_intdim,
                                      at::TensorOptions().dtype(torch::kFloat64));
    at::Tensor q = fixed_intcoord->vector_free2total(q_free);
    at::Tensor r = int2cart(q, init_guess_, intcoordset);
    at::Tensor e = compute_energy(r);
    energy = e[target_state].item<double>();
}

void energy_grad(double & energy, double * grad, const double * free_intgeom, const int32_t & free_intdim) {
    at::Tensor q_free = at::from_blob(const_cast<double *>(free_intgeom), free_intdim,
                                      at::TensorOptions().dtype(torch::kFloat64));
    at::Tensor q = fixed_intcoord->vector_free2total(q_free);
    at::Tensor r = int2cart(q, init_guess_, intcoordset);
    at::Tensor e, dHa;
    std::tie(e, dHa) = compute_energy_dHa(r);
    at::Tensor intgrad = intcoordset->gradient_cart2int(r, dHa[target_state][target_state]);
    at::Tensor free_intgrad = fixed_intcoord->vector_total2free(intgrad);
    energy = e[target_state].item<double>();
    std::memcpy(grad, free_intgrad.data_ptr<double>(), free_intdim * sizeof(double));
}

void Hessian(double * Hessian, const double * free_intgeom, const int32_t & free_intdim) {
    at::Tensor q_free = at::from_blob(const_cast<double *>(free_intgeom), free_intdim,
                                      at::TensorOptions().dtype(torch::kFloat64));
    at::Tensor q = fixed_intcoord->vector_free2total(q_free);
    at::Tensor r = int2cart(q, init_guess_, intcoordset);
    at::Tensor e, dHa;
    std::tie(e, dHa) = compute_energy_dHa(r);
    at::Tensor cartHess = compute_ddHa(r)[target_state][target_state];
    at::Tensor intHess = intcoordset->Hessian_cart2int(r, dHa[target_state][target_state], cartHess);
    at::Tensor free_intHess = fixed_intcoord->matrix_total2free(intHess);
    std::memcpy(Hessian, free_intHess.data_ptr<double>(), free_intdim * free_intdim * sizeof(double));
}

}

at::Tensor search_minimum(const at::Tensor & _init_guess) {
    init_guess_ = _init_guess;
    at::Tensor q = (*intcoordset)(_init_guess);
    at::Tensor q_free = fixed_intcoord->vector_total2free(q);
    Foptim::BFGS(energy, energy_grad, Hessian,
                 q_free.data_ptr<double>(), q_free.size(0),
                 20, 100, 1e-6, 1e-15);
    q = fixed_intcoord->vector_free2total(q_free);
    at::Tensor r = int2cart(q, _init_guess, intcoordset);
    return r;
}