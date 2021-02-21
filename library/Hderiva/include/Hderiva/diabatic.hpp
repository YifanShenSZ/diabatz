#ifndef Hderiva_diabatic_hpp
#define Hderiva_diabatic_hpp

#include <torch/torch.h>

#include <CppLibrary/utility.hpp>

namespace Hderiva {

at::Tensor DxHd(const at::Tensor & Hd, const at::Tensor & x,
const bool & create_graph = false);
// Assuming that Hd is computed from library *obnet*, `xs` are the input layers
// `JT` is the transposed Jacobian of the input layer over the coordinate
// This routine first computes the gradient of Hd over xs,
// then convert to d / dx * Hd
at::Tensor DxHd
(const at::Tensor & Hd, const CL::utility::matrix<at::Tensor> & xs,
const CL::utility::matrix<at::Tensor> & JTs,
const bool & create_graph = false);

// Assuming that Hd is computed from a neural network, cs = net.parameters()
// c = at::cat(cs)
at::Tensor DcHd(const at::Tensor & Hd, const std::vector<at::Tensor> & cs);

// Assuming that Hd is computed from a neural network, cs = net.parameters()
// c = at::cat(cs)
at::Tensor DcDxHd(const at::Tensor & DxHd, const std::vector<at::Tensor> & cs);

} // namespace Hderiva

#endif