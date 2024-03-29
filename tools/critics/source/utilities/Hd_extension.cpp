#include <tchem/linalg.hpp>

#include "../../include/global.hpp"

at::Tensor compute_ddHd(const at::Tensor & r) {
    const double dr = 1e-3;
    std::vector<at::Tensor> plus(r.size(0)), minus(r.size(0));
    #pragma omp parallel for
    for (size_t i = 0; i < r.size(0); i++) {
        at::Tensor energy;
        plus[i] = r.clone();
        plus[i][i] += dr;
        std::tie(energy, plus[i]) = HdKernel->compute_Hd_dHd(plus[i]);
        minus[i] = r.clone();
        minus[i][i] -= dr;
        std::tie(energy, minus[i]) = HdKernel->compute_Hd_dHd(minus[i]);
    }
    at::Tensor ddHd = r.new_empty({plus[0].size(0), plus[0].size(1), r.size(0), r.size(0)});
    for (size_t i = 0; i < r.size(0); i++) ddHd.select(2, i).copy_((plus[i] - minus[i]) / 2.0 / dr);
    return ddHd;
}

at::Tensor compute_energy(const at::Tensor & r) {
    at::Tensor Hd = (*HdKernel)(r);
    at::Tensor energy, states;
    std::tie(energy, states) = Hd.symeig();
    return energy;
}

std::tuple<at::Tensor, at::Tensor> compute_energy_dHa(const at::Tensor & r) {
    at::Tensor Hd, dHd;
    std::tie(Hd, dHd) = HdKernel->compute_Hd_dHd(r);
    at::Tensor energy, states;
    std::tie(energy, states) = Hd.symeig(true);
    at::Tensor dHa = tchem::linalg::UT_sy_U(dHd, states);
    return std::make_tuple(energy, dHa);
}

at::Tensor compute_ddHa(const at::Tensor & r) {
    // Here ddHa is ▽[(▽H)a], computed by finite difference of (▽H)a
    const double dr = 1e-3;
    std::vector<at::Tensor> plus(r.size(0)), minus(r.size(0));
    #pragma omp parallel for
    for (size_t i = 0; i < r.size(0); i++) {
        at::Tensor energy;
        plus[i] = r.clone();
        plus[i][i] += dr;
        std::tie(energy, plus[i]) = compute_energy_dHa(plus[i]);
        minus[i] = r.clone();
        minus[i][i] -= dr;
        std::tie(energy, minus[i]) = compute_energy_dHa(minus[i]);
    }
    at::Tensor ddHa = r.new_empty({plus[0].size(0), plus[0].size(1), r.size(0), r.size(0)});
    for (size_t i = 0; i < r.size(0); i++) ddHa.select(2, i).copy_((plus[i] - minus[i]) / 2.0 / dr);
    return ddHa;
}
