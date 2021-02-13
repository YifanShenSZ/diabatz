// This module assumes that Hd is computed from library *obnet*
// so xs are the input layers, cs are the parameters

#include <torch/torch.h>

#include <CppLibrary/utility.hpp>

namespace Hderiva {

CL::utility::matrix<at::Tensor> DxHd
(const at::Tensor & Hd, const CL::utility::matrix<at::Tensor> & xs,
const bool & create_graph) {
    assert(("Hd must be a matrix", Hd.sizes().size() == 2));
    assert(("Hd must be a square matrix", Hd.size(0) == Hd.size(1)));
    assert(("x must require gradient", xs[0][0].requires_grad()));
    CL::utility::matrix<at::Tensor> dHd(Hd.size(0));
    for (size_t i = 0; i < Hd.size(0); i++)
    for (size_t j = i; j < Hd.size(1); j++) {
        std::vector<at::Tensor> g = torch::autograd::grad({Hd[i][j]}, {xs[i][j]}, {}, true, create_graph);
        dHd[i][j] = g[0];
    }
    return dHd;
}

CL::utility::matrix<std::vector<at::Tensor>> DcHd
(const at::Tensor & Hd, const CL::utility::matrix<std::vector<at::Tensor>> & cs) {
    assert(("Hd must be a matrix", Hd.sizes().size() == 2));
    assert(("Hd must be a square matrix", Hd.size(0) == Hd.size(1)));
    CL::utility::matrix<std::vector<at::Tensor>> dHd(Hd.size(0));
    for (size_t i = 0; i < Hd.size(0); i++)
    for (size_t j = i; j < Hd.size(1); j++) {
        torch::autograd::variable_list g = torch::autograd::grad({Hd[i][j]}, {cs[i][j]}, {}, true);
        dHd[i][j] = g;
    }
    return dHd;
}

CL::utility::matrix<std::vector<at::Tensor>> DcDxHd
(const CL::utility::matrix<at::Tensor> & DxHd, const CL::utility::matrix<std::vector<at::Tensor>> & cs) {
    CL::utility::matrix<std::vector<at::Tensor>> ddHd(DxHd.size(0));
    for (size_t i = 0; i < DxHd.size(0); i++)
    for (size_t j = i; j < DxHd.size(1); j++)
    for (size_t k = 0; k < DxHd[i][j].size(0); k++) {
        torch::autograd::variable_list g = torch::autograd::grad({DxHd[i][j][k]}, {cs[i][j]}, {}, true);
        ddHd[i][j] = g;
    }
    return ddHd;
}

} // namespace Hderiva