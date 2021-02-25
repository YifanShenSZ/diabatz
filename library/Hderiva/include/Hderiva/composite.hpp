#ifndef Hderiva_composite_hpp
#define Hderiva_composite_hpp

#include <torch/torch.h>

namespace Hderiva {

// (d / dx * H)c = Ud^T. (d / dx * Hd) . Ud

// d / dc * Hc = Ud^T. (d / dc * Hd) . Ud
//             + [Hc, M]
at::Tensor DcHc
(const at::Tensor & Hc, const at::Tensor & DxHd, const at::Tensor & DcHd, const at::Tensor & DcDxHd,
const at::Tensor & eigval, const at::Tensor & eigvec);

// d / dc * (d / dx * H)c = Ud^T. (d / dc * d / dx * Hd) . Ud
//                        + [(d / dx * H)c, M]
at::Tensor DcDxHc
(const at::Tensor & DxHc, const at::Tensor & DxHd, const at::Tensor & DcDxHd,
const at::Tensor & eigval, const at::Tensor & eigvec);

std::tuple<at::Tensor, at::Tensor> DcHc_DcDxHc
(const at::Tensor & Hc, const at::Tensor & DxHc,
const at::Tensor & DxHd, const at::Tensor & DcHd, const at::Tensor & DcDxHd,
const at::Tensor & eigval, const at::Tensor & eigvec);

// dot product defined with a metric S
at::Tensor DcHc
(const at::Tensor & Hc, const at::Tensor & DxHd, const at::Tensor & DcHd, const at::Tensor & DcDxHd,
const at::Tensor & eigval, const at::Tensor & eigvec, const at::Tensor & S);
at::Tensor DcDxHc
(const at::Tensor & DxHc, const at::Tensor & DxHd, const at::Tensor & DcDxHd,
const at::Tensor & eigval, const at::Tensor & eigvec, const at::Tensor & S);
std::tuple<at::Tensor, at::Tensor> DcHc_DcDxHc
(const at::Tensor & Hc, const at::Tensor & DxHc,
const at::Tensor & DxHd, const at::Tensor & DcHd, const at::Tensor & DcDxHd,
const at::Tensor & eigval, const at::Tensor & eigvec, const at::Tensor & S);

} // namespace Hderiva

#endif