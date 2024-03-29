#include <torch/torch.h>

#include <CppLibrary/utility.hpp>

namespace Hderiva {

// Assuming that Hd is computed from library *DimRed* and *obnet*, `ls` are the input layers
// `JlrT` is the transposed Jacobian of the input layer over the reduced coordinate
// `JrxT` is the transposed Jacobian of the reduced coordinate over the coordinate
at::Tensor DxHd
(const at::Tensor & Hd, const CL::utility::matrix<at::Tensor> & ls,
const CL::utility::matrix<at::Tensor> & JlrTs, const at::Tensor & JrxT) {
    if (Hd.sizes().size() != 2) throw std::invalid_argument(
    "Hderiva::DxHd: Hd must be a matrix");
    if (Hd.size(0) != Hd.size(1)) throw std::invalid_argument(
    "Hderiva::DxHd: Hd must be a square matrix");
    for (size_t i = 0; i < ls.size(0); i++)
    for (size_t j = i; j < ls.size(1); j++)
    if (! ls[i][j].requires_grad()) throw std::invalid_argument(
    "Hderiva::DxHd: The input layers must require gradient");
    at::Tensor dHd = Hd.new_empty({Hd.size(0), Hd.size(1), JrxT.size(0)});
    for (size_t i = 0; i < Hd.size(0); i++)
    for (size_t j = i; j < Hd.size(1); j++) {
        auto g = torch::autograd::grad({Hd[i][j]}, {ls[i][j]}, {}, true);
        dHd[i][j].copy_(JrxT.mv(JlrTs[i][j].mv(g[0])));
    }
    return dHd;
}

// Assuming that Hd is computed from library *DimRed* and *obnet*, `ls` are the input layers, cs = obnet.parameters()
// `JlrT` is the transposed Jacobian of the input layer over the reduced coordinate
// `JrcT` is the transposed Jacobian of the reduced coordinate over DimRed.parameters()
// c = at::cat({at::cat(DimRed.parameters()), at::cat(obnet.parameters())})
at::Tensor DcHd(const at::Tensor & Hd, const CL::utility::matrix<at::Tensor> & ls, const std::vector<at::Tensor> & cs,
const CL::utility::matrix<at::Tensor> & JlrTs, const at::Tensor & JrcT) {
    if (Hd.sizes().size() != 2) throw std::invalid_argument(
    "Hderiva::DcHd: Hd must be a matrix");
    if (Hd.size(0) != Hd.size(1)) throw std::invalid_argument(
    "Hderiva::DcHd: Hd must be a square matrix");
    for (size_t i = 0; i < ls.size(0); i++)
    for (size_t j = i; j < ls.size(1); j++)
    if (! ls[i][j].requires_grad()) throw std::invalid_argument(
    "Hderiva::DcHd: The input layers must require gradient");
    int64_t Nc_DimRed = JrcT.size(0);
    int64_t Nc_obnet = 0;
    for (const at::Tensor & c : cs) Nc_obnet += c.numel();
    at::Tensor dHd = Hd.new_empty({Hd.size(0), Hd.size(1), Nc_DimRed + Nc_obnet});
    for (size_t i = 0; i < Hd.size(0); i++)
    for (size_t j = i; j < Hd.size(1); j++) {
        // over DimRed.parameters()
        auto g = torch::autograd::grad({Hd[i][j]}, {ls[i][j]}, {}, true);
        dHd[i][j].slice(0, 0, Nc_DimRed).copy_(JrcT.mv(JlrTs[i][j].mv(g[0])));
        // over obnet.parameters()
        auto gs = torch::autograd::grad({Hd[i][j]}, {cs}, {}, true, false, true);
        for (size_t l = 0; l < cs.size(); l++) {
            if (! gs[l].defined()) gs[l] = cs[l].new_zeros(cs[l].sizes());
            if (gs[l].sizes().size() != 1) gs[l] = gs[l].view(gs[l].numel());
        }
        dHd[i][j].slice(0, Nc_DimRed).copy_(at::cat(gs));
    }
    return dHd;
}

// Assuming that Hd is computed from library *DimRed* and *obnet*, `ls` are the input layers, cs = obnet.parameters()
// `JlrT` is the transposed Jacobian of the input layer over the reduced coordinate
// `Klr` is the 2nd-order Jacobian of the input layer over the reduced coordinate
// `JrxT` is the transposed Jacobian of the reduced coordinate over the coordinate
// `JrcT` is the transposed Jacobian of the reduced coordinate over DimRed.parameters()
// `Krxc` is the 2nd-order Jacobian of the reduced coordinate over the coordinate and DimRed.parameters()
// c = at::cat({at::cat(DimRed.parameters()), at::cat(obnet.parameters())})
at::Tensor DcDxHd(const at::Tensor & Hd, const CL::utility::matrix<at::Tensor> & ls, const std::vector<at::Tensor> & cs,
const CL::utility::matrix<at::Tensor> & JlrTs, const CL::utility::matrix<at::Tensor> & Klrs, 
const at::Tensor & JrxT, const at::Tensor & JrcT, const at::Tensor & Krxc) {
    if (Hd.sizes().size() != 2) throw std::invalid_argument(
    "Hderiva::DcHd: Hd must be a matrix");
    if (Hd.size(0) != Hd.size(1)) throw std::invalid_argument(
    "Hderiva::DcHd: Hd must be a square matrix");
    for (size_t i = 0; i < ls.size(0); i++)
    for (size_t j = i; j < ls.size(1); j++)
    if (! ls[i][j].requires_grad()) throw std::invalid_argument(
    "Hderiva::DcDxHd: The input layers must require gradient");
    // Prepare
    at::Tensor Jrc = JrcT.transpose(0, 1);
    int64_t Nc_DimRed = Krxc.size(2);
    int64_t Nc_obnet = 0;
    for (const at::Tensor & c : cs) Nc_obnet += c.numel();
    // Backward propagation and transformation
    at::Tensor ddHd = Hd.new_empty({Hd.size(0), Hd.size(1), JrxT.size(0), Nc_DimRed + Nc_obnet});
    for (size_t i = 0; i < Hd.size(0); i++)
    for (size_t j = i; j < Hd.size(1); j++) {
        const at::Tensor & JlrT = JlrTs[i][j];
        at::Tensor Jlr = JlrT.transpose(0, 1);
        std::vector<at::Tensor> g = torch::autograd::grad({Hd[i][j]}, {ls[i][j]}, {}, true, true);
        const at::Tensor & DlHdij = g[0];
        // over DimRed.parameters()
        ddHd[i][j].slice(1, 0, Nc_DimRed).copy_(
            JrxT.mm(at::matmul(DlHdij, Klrs[i][j].transpose(0, 1))).mm(Jrc)
          + at::matmul(JlrT.mv(DlHdij), Krxc.transpose(0, 1))
        );
        at::Tensor DlDlHdij = DlHdij.new_empty({DlHdij.size(0), DlHdij.size(0)});
        for (size_t k = 0; k < DlHdij.size(0); k++) {
            auto g = torch::autograd::grad({DlHdij[k]}, {ls[i][j]}, {}, true, false, true);
            if (g[0].defined()) DlDlHdij[k].copy_(g[0]);
            else DlDlHdij[k].zero_();
        }
        ddHd[i][j].slice(1, 0, Nc_DimRed) += JrxT.mm(JlrT.mm(DlDlHdij).mm(Jlr)).mm(Jrc);
        // over obnet.parameters()
        at::Tensor DcDlHdij = DlHdij.new_empty({DlHdij.size(0), Nc_obnet});
        for (size_t k = 0; k < DlHdij.size(0); k++) {
            auto gs = torch::autograd::grad({DlHdij[k]}, {cs}, {}, true, false, true);
            for (size_t l = 0; l < cs.size(); l++) {
                if (! gs[l].defined()) gs[l] = cs[l].new_zeros(cs[l].sizes());
                if (gs[l].sizes().size() != 1) gs[l] = gs[l].view(gs[l].numel());
            }
            DcDlHdij[k].copy_(at::cat(gs));
        }
        ddHd[i][j].slice(1, Nc_DimRed).copy_(JrxT.mm(JlrT.mm(DcDlHdij)));
    }
    return ddHd;
}

}