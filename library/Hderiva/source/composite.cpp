#include <tchem/linalg.hpp>

#include <Hderiva/basic.hpp>

namespace Hderiva {

// (d / dx * H)c = Ud^T. (d / dx * Hd) . Ud

// d / dc * Hc = Ud^T. (d / dc * Hd) . Ud
//             + [Hc, M]
at::Tensor DcHc
(const at::Tensor & Hc, const at::Tensor & DxHd, const at::Tensor & DcHd, const at::Tensor & DcDxHd,
const at::Tensor & eigval, const at::Tensor & eigvec) {
    at::Tensor DcO = tchem::linalg::sy4matmvmulsy3(DcDxHd.transpose(-1, -2), DxHd);
    DcO = DcO + DcO.transpose(0, 1);
    at::Tensor nac = tchem::linalg::UT_sy_U(DcO, eigvec);
    for (size_t i = 0    ; i < eigval.size(0); i++)
    for (size_t j = i + 1; j < eigval.size(0); j++)
    nac[i][j] /= eigval[j] - eigval[i];
    at::Tensor DcHc = tchem::linalg::UT_sy_U(DcHd, eigvec)
                    + commutor_term(Hc, nac);
    return DcHc;
}

// d / dc * (d / dx * H)c = Ud^T. (d / dc * d / dx * Hd) . Ud
//                        + [(d / dx * H)c, M]
at::Tensor DcDxHc
(const at::Tensor & DxHc, const at::Tensor & DxHd, const at::Tensor & DcDxHd,
const at::Tensor & eigval, const at::Tensor & eigvec) {
    at::Tensor DcO = tchem::linalg::sy4matmvmulsy3(DcDxHd.transpose(-1, -2), DxHd);
    DcO = DcO + DcO.transpose(0, 1);
    at::Tensor nac = tchem::linalg::UT_sy_U(DcO, eigvec);
    for (size_t i = 0    ; i < eigval.size(0); i++)
    for (size_t j = i + 1; j < eigval.size(0); j++)
    nac[i][j] /= eigval[j] - eigval[i];
    at::Tensor DcDxHc = tchem::linalg::UT_sy_U(DcDxHd, eigvec)
                      + commutor_term(DxHc, nac);
    return DcDxHc;
}

std::tuple<at::Tensor, at::Tensor> DcHc_DcDxHc
(const at::Tensor & Hc, const at::Tensor & DxHc,
const at::Tensor & DxHd, const at::Tensor & DcHd, const at::Tensor & DcDxHd,
const at::Tensor & eigval, const at::Tensor & eigvec) {
    at::Tensor DcO = tchem::linalg::sy4matmvmulsy3(DcDxHd.transpose(-1, -2), DxHd);
    DcO = DcO + DcO.transpose(0, 1);
    at::Tensor nac = tchem::linalg::UT_sy_U(DcO, eigvec);
    for (size_t i = 0    ; i < eigval.size(0); i++)
    for (size_t j = i + 1; j < eigval.size(0); j++)
    nac[i][j] /= eigval[j] - eigval[i];
    at::Tensor DcHc = tchem::linalg::UT_sy_U(DcHd, eigvec)
                    + commutor_term(Hc, nac);
    at::Tensor DcDxHc = tchem::linalg::UT_sy_U(DcDxHd, eigvec)
                      + commutor_term(DxHc, nac);
    return std::make_tuple(DcHc, DcDxHc);
}

// dot product defined with a metric S
at::Tensor DcHc
(const at::Tensor & Hc, const at::Tensor & DxHd, const at::Tensor & DcHd, const at::Tensor & DcDxHd,
const at::Tensor & eigval, const at::Tensor & eigvec, const at::Tensor & S) {
    at::Tensor DcO = tchem::linalg::sy4matmvmulsy3(DcDxHd.transpose(-1, -2), DxHd, S);
    DcO = DcO + DcO.transpose(0, 1);
    at::Tensor nac = tchem::linalg::UT_sy_U(DcO, eigvec);
    for (size_t i = 0    ; i < eigval.size(0); i++)
    for (size_t j = i + 1; j < eigval.size(0); j++)
    nac[i][j] /= eigval[j] - eigval[i];
    at::Tensor DcHc = tchem::linalg::UT_sy_U(DcHd, eigvec)
                    + commutor_term(Hc, nac);
    return DcHc;
}
at::Tensor DcDxHc
(const at::Tensor & DxHc, const at::Tensor & DxHd, const at::Tensor & DcDxHd,
const at::Tensor & eigval, const at::Tensor & eigvec, const at::Tensor & S) {
    at::Tensor DcO = tchem::linalg::sy4matmvmulsy3(DcDxHd.transpose(-1, -2), DxHd, S);
    DcO = DcO + DcO.transpose(0, 1);
    at::Tensor nac = tchem::linalg::UT_sy_U(DcO, eigvec);
    for (size_t i = 0    ; i < eigval.size(0); i++)
    for (size_t j = i + 1; j < eigval.size(0); j++)
    nac[i][j] /= eigval[j] - eigval[i];
    at::Tensor DcDxHc = tchem::linalg::UT_sy_U(DcDxHd, eigvec)
                      + commutor_term(DxHc, nac);
    return DcDxHc;
}
std::tuple<at::Tensor, at::Tensor> DcHc_DcDxHc
(const at::Tensor & Hc, const at::Tensor & DxHc,
const at::Tensor & DxHd, const at::Tensor & DcHd, const at::Tensor & DcDxHd,
const at::Tensor & eigval, const at::Tensor & eigvec, const at::Tensor & S) {
    at::Tensor DcO = tchem::linalg::sy4matmvmulsy3(DcDxHd.transpose(-1, -2), DxHd, S);
    DcO = DcO + DcO.transpose(0, 1);
    at::Tensor nac = tchem::linalg::UT_sy_U(DcO, eigvec);
    for (size_t i = 0    ; i < eigval.size(0); i++)
    for (size_t j = i + 1; j < eigval.size(0); j++)
    nac[i][j] /= eigval[j] - eigval[i];
    at::Tensor DcHc = tchem::linalg::UT_sy_U(DcHd, eigvec)
                    + commutor_term(Hc, nac);
    at::Tensor DcDxHc = tchem::linalg::UT_sy_U(DcDxHd, eigvec)
                      + commutor_term(DxHc, nac);
    return std::make_tuple(DcHc, DcDxHc);
}

} // namespace Hderiva