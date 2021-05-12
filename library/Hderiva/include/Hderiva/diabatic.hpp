#ifndef Hderiva_diabatic_hpp
#define Hderiva_diabatic_hpp

#include <torch/torch.h>

#include <CppLibrary/utility.hpp>

namespace Hderiva {

at::Tensor DxHd(const at::Tensor & Hd, const at::Tensor & x,
const bool & create_graph = false);
// Assuming that Hd is computed from library *obnet*, `ls` are the input layers
// `JT` is the transposed Jacobian of the input layer over the coordinate
at::Tensor DxHd
(const at::Tensor & Hd, const CL::utility::matrix<at::Tensor> & ls,
const CL::utility::matrix<at::Tensor> & JTs,
const bool & create_graph = false);
// Assuming that Hd is computed from library *obnet*, `ls` are the input layers
// `JlrT` is the transposed Jacobian of the input layer over the reduced coordinate
// `JrxT` is the transposed Jacobian of the reduced coordinate over the coordinate
at::Tensor DxHd
(const at::Tensor & Hd, const CL::utility::matrix<at::Tensor> & ls,
const CL::utility::matrix<at::Tensor> & JlrTs, const at::Tensor & JrxT);

// Assuming that Hd is computed from a neural network, cs = net.parameters()
// c = at::cat(cs)
at::Tensor DcHd(const at::Tensor & Hd, const std::vector<at::Tensor> & cs);
// Assuming that Hd is computed from library *DimRed* and *obnet*, `ls` are the input layers, cs = obnet.parameters()
// `JlrT` is the transposed Jacobian of the input layer over the reduced coordinate
// `JrcT` is the transposed Jacobian of the reduced coordinate over DimRed.parameters()
// c = at::cat({at::cat(DimRed.parameters()), at::cat(obnet.parameters())})
at::Tensor DcHd(const at::Tensor & Hd, const CL::utility::matrix<at::Tensor> & ls, const std::vector<at::Tensor> & cs,
const CL::utility::matrix<at::Tensor> & JlrTs, const at::Tensor & JrcT);

// Assuming that Hd is computed from a neural network, cs = net.parameters()
// c = at::cat(cs)
at::Tensor DcDxHd(const at::Tensor & DxHd, const std::vector<at::Tensor> & cs);
// Assuming that Hd is computed from library *DimRed* and *obnet*, `ls` are the input layers, cs = obnet.parameters()
// `JlrT` is the transposed Jacobian of the input layer over the reduced coordinate
// `JrcT` is the transposed Jacobian of the reduced coordinate over DimRed.parameters()
// `KrxcT` is the transposed 2nd-order Jacobian of the reduced coordinate over the coordinate and DimRed.parameters()
//            The transpose is performed between r and x
// c = at::cat({at::cat(DimRed.parameters()), at::cat(obnet.parameters())})
at::Tensor DcDxHd(const at::Tensor & Hd, const CL::utility::matrix<at::Tensor> & ls, const std::vector<at::Tensor> & cs,
const CL::utility::matrix<at::Tensor> & JlrTs, const at::Tensor & JrxT, const at::Tensor & JrcT, const at::Tensor & KrxcT);

} // namespace Hderiva

#endif