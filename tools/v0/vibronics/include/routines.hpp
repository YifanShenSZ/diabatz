#ifndef routines_hpp
#define routines_hpp

#include <tchem/chemistry.hpp>

#include <Hd/kernel.hpp>

at::Tensor read_Columbus(const at::Tensor & r, const std::string & hessian_file);

std::vector<at::Tensor> Hessian_cart2int(
const at::Tensor & r, const std::vector<size_t> & CNPI2point, const at::Tensor & carthess);

at::Tensor compute_intddHd(const at::Tensor & r, const Hd::kernel & Hdkernel);

void final2init(
const std::vector<at::Tensor> & init_qs, const std::vector<at::Tensor> & final_qs,
const tchem::chem::SANormalMode & init_vib, const tchem::chem::SANormalMode & final_vib);

void suggest_phonons(const double & contour,
const std::vector<at::Tensor> & init_qs, const std::vector<at::Tensor> & final_qs,
const tchem::chem::SANormalMode & init_vib, const tchem::chem::SANormalMode & final_vib);

void int2normal(const Hd::kernel & Hdkernel, const tchem::chem::SANormalMode & final_vib);

#endif