#include <torch/torch.h>

#include <CppLibrary/utility.hpp>

namespace Hderiva {

at::Tensor DxHd(const at::Tensor & Hd, const at::Tensor & x,
const bool & create_graph = false) {
    if (Hd.sizes().size() != 2) throw std::invalid_argument(
    "Hderiva::DxHd: Hd must be a matrix");
    if (Hd.size(0) != Hd.size(1)) throw std::invalid_argument(
    "Hderiva::DxHd: Hd must be a square matrix");
    if (! x.requires_grad()) throw std::invalid_argument(
    "Hderiva::DxHd: x must require gradient");
    at::Tensor dHd = Hd.new_empty({Hd.size(0), Hd.size(1), x.size(0)});
    for (size_t i = 0; i < Hd.size(0); i++)
    for (size_t j = i; j < Hd.size(1); j++) {
        std::vector<at::Tensor> g = torch::autograd::grad({Hd[i][j]}, {x}, {}, true, create_graph);
        dHd[i][j] = g[0];
    }
    return dHd;
}
// Assuming that Hd is computed from library *obnet*, `xs` are the input layers
// `JT` is the transposed Jacobian of the input layer over the coordinate
// This routine first computes the gradient of Hd over xs,
// then convert to d / dx * Hd
at::Tensor DxHd
(const at::Tensor & Hd, const CL::utility::matrix<at::Tensor> & xs,
const CL::utility::matrix<at::Tensor> & JTs,
const bool & create_graph = false) {
    if (Hd.sizes().size() != 2) throw std::invalid_argument(
    "Hderiva::DxHd: Hd must be a matrix");
    if (Hd.size(0) != Hd.size(1)) throw std::invalid_argument(
    "Hderiva::DxHd: Hd must be a square matrix");
    for (size_t i = 0; i < xs.size(0); i++)
    for (size_t j = i; j < xs.size(1); j++)
    if (! xs[i][j].requires_grad()) throw std::invalid_argument(
    "Hderiva::DxHd: The input layers must require gradient");
    at::Tensor dHd = Hd.new_empty({Hd.size(0), Hd.size(1), JTs[0][0].size(0)});
    for (size_t i = 0; i < Hd.size(0); i++)
    for (size_t j = i; j < Hd.size(1); j++) {
        std::vector<at::Tensor> g = torch::autograd::grad({Hd[i][j]}, {xs[i][j]}, {}, true, create_graph);
        dHd[i][j] = JTs[i][j].mv(g[0]);
    }
    return dHd;
}

// Assuming that Hd is computed from a neural network, cs = net.parameters()
// c = at::cat(cs)
at::Tensor DcHd(const at::Tensor & Hd, const std::vector<at::Tensor> & cs) {
    if (Hd.sizes().size() != 2) throw std::invalid_argument(
    "Hderiva::DcHd: Hd must be a matrix");
    if (Hd.size(0) != Hd.size(1)) throw std::invalid_argument(
    "Hderiva::DcHd: Hd must be a square matrix");
    int64_t NPars = 0;
    for (const at::Tensor & c : cs) NPars += c.numel();
    at::Tensor dHd = Hd.new_empty({Hd.size(0), Hd.size(1), NPars});
    for (size_t i = 0; i < Hd.size(0); i++)
    for (size_t j = i; j < Hd.size(1); j++) {
        torch::autograd::variable_list gs = torch::autograd::grad({Hd[i][j]}, {cs}, {}, true, false, true);
        for (size_t l = 0; l < cs.size(); l++) {
            if (! gs[l].defined()) gs[l] = cs[l].new_zeros(cs[l].sizes());
            if (gs[l].sizes().size() != 1) gs[l] = gs[l].view(gs[l].numel());
        }
        dHd[i][j] = at::cat(gs);
    }
    return dHd;
}

// Assuming that Hd is computed from a neural network, cs = net.parameters()
// c = at::cat(cs)
at::Tensor DcDxHd
(const at::Tensor & DxHd, const std::vector<at::Tensor> & cs) {
    if (DxHd.sizes().size() != 3) throw std::invalid_argument(
    "Hderiva::DcDxHd: DxHd must be a 3rd-order tensor");
    if (DxHd.size(0) != DxHd.size(1)) throw std::invalid_argument(
    "Hderiva::DcDxHd: The matrix part of DxHd must be square");
    int64_t NPars = 0;
    for (const at::Tensor & c : cs) NPars += c.numel();
    at::Tensor ddHd = DxHd.new_empty({DxHd.size(0), DxHd.size(1), DxHd.size(2), NPars});
    for (size_t i = 0; i < DxHd.size(0); i++)
    for (size_t j = i; j < DxHd.size(1); j++)
    for (size_t k = 0; k < DxHd.size(2); k++) {
        torch::autograd::variable_list gs = torch::autograd::grad({DxHd[i][j][k]}, {cs}, {}, true, false, true);
        for (size_t l = 0; l < cs.size(); l++) {
            if (! gs[l].defined()) gs[l] = cs[l].new_zeros(cs[l].sizes());
            if (gs[l].sizes().size() != 1) gs[l] = gs[l].view(gs[l].numel());
        }
        ddHd[i][j][k] = at::cat(gs);
    }
    return ddHd;
}

} // namespace Hderiva