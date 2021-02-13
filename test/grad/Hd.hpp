#ifndef Hd_hpp
#define Hd_hpp

#include <torch/torch.h>

namespace libHd {

at::Tensor Hd(const at::Tensor & c00, const at::Tensor & c01, const at::Tensor & c11, const at::Tensor & q);

// d / dq of Hd matrix elements
// Because of diabaticity, this is equivalent to
// the matrix form of operator d / dq * H in diabatic representation
at::Tensor DqHd(const at::Tensor & c00, const at::Tensor & c01, const at::Tensor & c11, const at::Tensor & q);
// Analytical version
at::Tensor analytical_DqHd(const at::Tensor & c00, const at::Tensor & c01, const at::Tensor & c11, const at::Tensor & q);

// d / dc of Hd matrix elements
// Because of diabaticity, this is equivalent to
// the matrix form of operator d / dc * H in diabatic representation
std::tuple<at::Tensor, at::Tensor, at::Tensor> DcHd
(const at::Tensor & c00, const at::Tensor & c01, const at::Tensor & c11, const at::Tensor & q);
// Analytical version
std::tuple<at::Tensor, at::Tensor, at::Tensor> analytical_DcHd
(const at::Tensor & c00, const at::Tensor & c01, const at::Tensor & c11, const at::Tensor & q);

// d / dc * d / dq of Hd matrix elements
// Because of diabaticity, this is equivalent to
// the matrix form of operator d / dc * d / dq * H in diabatic representation
std::tuple<at::Tensor, at::Tensor, at::Tensor> DcDqHd
(const at::Tensor & c00, const at::Tensor & c01, const at::Tensor & c11, const at::Tensor & q);
// Analytical version
std::tuple<at::Tensor, at::Tensor, at::Tensor> analytical_DcDqHd
(const at::Tensor & c00, const at::Tensor & c01, const at::Tensor & c11, const at::Tensor & q);





at::Tensor energies(const at::Tensor & c00, const at::Tensor & c01, const at::Tensor & c11, const at::Tensor & q);

// The matrix form of operator d / dq * H in adiabatic representation
at::Tensor DqH_a(const at::Tensor & c00, const at::Tensor & c01, const at::Tensor & c11, const at::Tensor & q);





at::Tensor Hc(const at::Tensor & c00, const at::Tensor & c01, const at::Tensor & c11, const at::Tensor & q);

// The matrix form of operator d / dq * H in composite representation
at::Tensor DqH_c(const at::Tensor & c00, const at::Tensor & c01, const at::Tensor & c11, const at::Tensor & q);

} // namespace libHd

#endif