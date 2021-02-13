// This module assumes that Hd is computed from library *obnet*
// so xs are the input layers, cs are the parameters

#include <torch/torch.h>

#include <CppLibrary/utility.hpp>

namespace Hderiva {

CL::utility::matrix<at::Tensor> DxHd
(const at::Tensor & Hd, const CL::utility::matrix<at::Tensor> & xs,
const bool & create_graph = false);

CL::utility::matrix<std::vector<at::Tensor>> DcHd
(const at::Tensor & Hd, const CL::utility::matrix<std::vector<at::Tensor>> & cs);

CL::utility::matrix<std::vector<at::Tensor>> DcDxHd
(const CL::utility::matrix<at::Tensor> & DxHd, const CL::utility::matrix<std::vector<at::Tensor>> & cs);

} // namespace Hderiva