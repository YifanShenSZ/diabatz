#include <tchem/linalg.hpp>

#include <Hderiva/basic.hpp>

namespace Hderiva {

// (d / dx * H)c = Ud^T. (d / dx * Hd) . Ud

// d / dc * Hc = Ud^T. (d / dc * Hd) . Ud
//             + [Hc, M]
at::Tensor DcHc
(const at::Tensor & Hc, const at::Tensor & DxHd, const at::Tensor & DcHd, const at::Tensor & DcDxHd,
const at::Tensor & eigvals, const at::Tensor & eigvecs) {
    at::Tensor DcO = tchem::LA::sy4matmvmulsy3(DcDxHd.transpose_(-1, -2), DxHd);
    DcO = DcO + DcO.transpose(0, 1);
    at::Tensor nac = tchem::LA::UT_sy_U(DcO, eigvecs);
    for (size_t i = 0    ; i < eigvals.size(0); i++)
    for (size_t j = i + 1; j < eigvals.size(0); j++)
    nac[i][j] /= eigvals[j] - eigvals[i];
    at::Tensor DcHc = tchem::LA::UT_sy_U(DcHd, eigvecs)
                    + commutor_term(Hc, nac);
    return DcHc;
}

// d / dc * (d / dx * H)c = Ud^T. (d / dc * d / dx * Hd) . Ud
//                        + [(d / dx * H)c, M]
at::Tensor DcDxHc
(const at::Tensor & DxHc, const at::Tensor & DxHd, const at::Tensor & DcDxHd,
const at::Tensor & eigvals, const at::Tensor & eigvecs) {
    at::Tensor DcO = tchem::LA::sy4matmvmulsy3(DcDxHd.transpose_(-1, -2), DxHd);
    DcO = DcO + DcO.transpose(0, 1);
    at::Tensor nac = tchem::LA::UT_sy_U(DcO, eigvecs);
    for (size_t i = 0    ; i < eigvals.size(0); i++)
    for (size_t j = i + 1; j < eigvals.size(0); j++)
    nac[i][j] /= eigvals[j] - eigvals[i];
    at::Tensor DcDxHc = tchem::LA::UT_sy_U(DcDxHd, eigvecs)
                      + commutor_term(DxHc, nac);
    return DcDxHc;
}

std::tuple<at::Tensor, at::Tensor> DcHc_DcDxHc
(const at::Tensor & Hc, const at::Tensor & DxHc,
const at::Tensor & DxHd, const at::Tensor & DcHd, const at::Tensor & DcDxHd,
const at::Tensor & eigvals, const at::Tensor & eigvecs) {
    at::Tensor DcO = tchem::LA::sy4matmvmulsy3(DcDxHd.transpose_(-1, -2), DxHd);
    DcO = DcO + DcO.transpose(0, 1);
    at::Tensor nac = tchem::LA::UT_sy_U(DcO, eigvecs);
    for (size_t i = 0    ; i < eigvals.size(0); i++)
    for (size_t j = i + 1; j < eigvals.size(0); j++)
    nac[i][j] /= eigvals[j] - eigvals[i];
    at::Tensor DcHc = tchem::LA::UT_sy_U(DcHd, eigvecs)
                    + commutor_term(Hc, nac);
    at::Tensor DcDxHc = tchem::LA::UT_sy_U(DcDxHd, eigvecs)
                      + commutor_term(DxHc, nac);
    return std::make_tuple(DcHc, DcDxHc);
}

} // namespace Hderiva