#include <torch/torch.h>

#include <CppLibrary/utility.hpp>

namespace Hderiva {

// Assuming that Hd is computed from library *obnet*, `ls` are the input layers
// `JT` is the transposed Jacobian of the input layer over the coordinate
at::Tensor DxHd
(const at::Tensor & Hd, const CL::utility::matrix<at::Tensor> & ls,
const CL::utility::matrix<at::Tensor> & JTs,
const bool & create_graph = false) {
    if (Hd.sizes().size() != 2) throw std::invalid_argument(
    "Hderiva::DxHd: Hd must be a matrix");
    if (Hd.size(0) != Hd.size(1)) throw std::invalid_argument(
    "Hderiva::DxHd: Hd must be a square matrix");
    for (size_t i = 0; i < ls.size(0); i++)
    for (size_t j = i; j < ls.size(1); j++)
    if (! ls[i][j].requires_grad()) throw std::invalid_argument(
    "Hderiva::DxHd: The input layers must require gradient");
    at::Tensor dHd = Hd.new_empty({Hd.size(0), Hd.size(1), JTs[0][0].size(0)});
    for (size_t i = 0; i < Hd.size(0); i++)
    for (size_t j = i; j < Hd.size(1); j++) {
        auto g = torch::autograd::grad({Hd[i][j]}, {ls[i][j]}, {}, true, create_graph);
        dHd[i][j] = JTs[i][j].mv(g[0]);
    }
    return dHd;
}

// DcHd is the same

// Q: Why not a specialized DcDxHd?
// A: In that way we would backward through d / dls * Hd elements,
//    but ls usually have magnitudes more elements than x

}