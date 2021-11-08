#include "../include/fixed_intcoord.hpp"

Fixed_intcoord::Fixed_intcoord() {}
Fixed_intcoord::Fixed_intcoord(const int64_t & _intdim, const std::vector<size_t> & _fixed_coords, const at::Tensor & init_q)
: intdim_(_intdim), fixed_coords_(_fixed_coords) {
    // create free coordinates
    for (size_t i = 0; i < _intdim; i++)
    if (std::find(_fixed_coords.begin(), _fixed_coords.end(), i) == _fixed_coords.end())
    free_coords_.push_back(i);
    // get fixed values
    for (const size_t & fixed_coord : _fixed_coords)
    fixed_values_.push_back(init_q[fixed_coord].item<double>());
}
Fixed_intcoord::~Fixed_intcoord() {}

at::Tensor Fixed_intcoord::vector_free2total(const at::Tensor & V_free) const {
    at::Tensor V = V_free.new_empty(intdim_);
    for (size_t i = 0; i < fixed_coords_.size(); i++) V[fixed_coords_[i]].fill_(fixed_values_[i]);
    for (size_t i = 0; i <  free_coords_.size(); i++) V[ free_coords_[i]].copy_(V_free[i]);
    return V;
}
at::Tensor Fixed_intcoord::vector_total2free(const at::Tensor & V) const {
    at::Tensor V_free = V.new_empty(free_coords_.size());
    for (size_t i = 0; i < free_coords_.size(); i++) V_free[i].copy_(V[free_coords_[i]]);
    return V_free;
}

at::Tensor Fixed_intcoord::matrix_total2free(const at::Tensor & M) const {
    int64_t NFree = free_coords_.size();
    at::Tensor M_free = M.new_empty({NFree, NFree});
    for (size_t i = 0; i < NFree; i++)
    for (size_t j = 0; j < NFree; j++)
    M_free[i][j].copy_(M[free_coords_[i]][free_coords_[j]]);
    return M_free;
}