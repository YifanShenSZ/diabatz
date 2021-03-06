#include <tchem/linalg.hpp>

#include <Hderiva/basic.hpp>

namespace Hderiva {

// (d / dx * H)a = Ud^T. (d / dx * Hd) . Ud

// d / dc * E = (Ud^T. (d / dc * Hd) . Ud).diag(), by Hellmann–Feynman theorem

// d / dc * (d / dx * H)a = Ud^T. (d / dc * d / dx * Hd) . Ud
//                        + [(d / dx * H)a, M]
at::Tensor DcDxHa
(const at::Tensor & DxHa, const at::Tensor & DcHd, const at::Tensor & DcDxHd,
const at::Tensor & energy, const at::Tensor & states) {
    at::Tensor nac = tchem::linalg::UT_sy_U(DcHd, states);
    for (size_t i = 0    ; i < energy.size(0); i++)
    for (size_t j = i + 1; j < energy.size(0); j++)
    nac[i][j] /= energy[j] - energy[i];
    at::Tensor DcDxHa = tchem::linalg::UT_sy_U(DcDxHd, states)
                      + commutor_term(DxHa, nac);
    return DcDxHa;
}

} // namespace Hderiva