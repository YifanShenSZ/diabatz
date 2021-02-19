#include <torch/torch.h>

#include <CppLibrary/utility.hpp>

namespace Hderiva {

at::Tensor DxHd(const at::Tensor & Hd, const at::Tensor & x,
const bool & create_graph = false) {
    assert(("Hd must be a matrix", Hd.sizes().size() == 2));
    assert(("Hd must be a square matrix", Hd.size(0) == Hd.size(1)));
    assert(("x must require gradient", x.requires_grad()));
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
    assert(("Hd must be a matrix", Hd.sizes().size() == 2));
    assert(("Hd must be a square matrix", Hd.size(0) == Hd.size(1)));
    for (size_t i = 0; i < xs.size(0); i++)
    for (size_t j = i; j < xs.size(1); j++)
    assert(("The input layers must require gradient", xs[i][j].requires_grad()));
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
    assert(("Hd must be a matrix", Hd.sizes().size() == 2));
    assert(("Hd must be a square matrix", Hd.size(0) == Hd.size(1)));
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
    assert(("DxHd must be a 3rd-order tensor", DxHd.sizes().size() == 3));
    assert(("The matrix part of DxHd must be square", DxHd.size(0) == DxHd.size(1)));
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