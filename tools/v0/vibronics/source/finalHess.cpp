#include <Hd/kernel.hpp>

#include "../include/CNPI.hpp"

at::Tensor compute_intddHd(const at::Tensor & r, const Hd::kernel & Hdkernel) {
    const double dq = 1e-3;
    std::vector<at::Tensor> qs, Js;
    std::tie(qs, Js) = cart2CNPI(r);
    int64_t intdim = sasicset->intdim();
    // Finite difference
    std::vector<at::Tensor> plus(intdim), minus(intdim);
    #pragma omp parallel for
    for (size_t coord = 0; coord < intdim; coord++) {
        at::Tensor Hd, dHd;
        std::vector<at::Tensor> finite_qs(qs.size());
        // Locate which coordinate are we dealing with
        size_t irred, index = coord;
        for (irred = 0; irred < qs.size(); irred++)
        if (index < qs[irred].size(0)) break;
        else index -= qs[irred].size(0);
        // Compute SASIC ▽Hd
        for (size_t i = 0; i < qs.size(); i++) finite_qs[i] = qs[i].clone();
        finite_qs[irred][index] += dq;
        std::tie(Hd, plus[coord]) = Hdkernel.compute_Hd_dHd(finite_qs);
        for (size_t i = 0; i < qs.size(); i++) finite_qs[i] = qs[i].clone();
        finite_qs[irred][index] -= dq;
        std::tie(Hd, minus[coord]) = Hdkernel.compute_Hd_dHd(finite_qs);
    }
    at::Tensor ddHd = r.new_empty({plus[0].size(0), plus[0].size(1), intdim, intdim});
    for (size_t i = 0; i < intdim; i++) ddHd.select(2, i).copy_((plus[i] - minus[i]) / 2.0 / dq);
    return ddHd;
}