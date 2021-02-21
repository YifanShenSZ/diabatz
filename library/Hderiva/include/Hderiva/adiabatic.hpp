#ifndef Hderiva_adiabatic_hpp
#define Hderiva_adiabatic_hpp

#include <torch/torch.h>

namespace Hderiva {

// (d / dx * H)a = Ud^T. (d / dx * Hd) . Ud

// d / dc * E = (Ud^T. (d / dc * Hd) . Ud).diag(), by Hellmann–Feynman theorem

// d / dc * (d / dx * H)a = Ud^T. (d / dc * d / dx * Hd) . Ud
//                        + [(d / dx * H)a, M]
at::Tensor DcDxHa
(const at::Tensor & DxHa, const at::Tensor & DcHd, const at::Tensor & DcDxHd,
const at::Tensor & energies, const at::Tensor & states);

} // namespace Hderiva

#endif