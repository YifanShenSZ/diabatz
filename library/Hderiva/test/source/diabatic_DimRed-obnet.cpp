#include <abinitio/SAreader.hpp>

#include <DimRed/encoder.hpp>
#include <DimRed/decoder.hpp>

#include <Hderiva/diabatic.hpp>

#include "../include/global.hpp"

at::Tensor compute_Jrq(const at::Tensor & r, const at::Tensor & q) {
    at::Tensor J = r.new_empty({r.size(0), q.size(0)});
    for (size_t i = 0; i < J.size(0); i++) {
        auto g = torch::autograd::grad({r[i]}, {q}, {}, true);
        J[i].copy_(g[0]);
    }
    return J;
}

at::Tensor compute_Jrc(const at::Tensor & r, const std::vector<at::Tensor> & cs) {
    int64_t NPars = 0;
    for (const at::Tensor & c : cs) NPars += c.numel();
    at::Tensor J = r.new_empty({r.size(0), NPars});
    for (size_t i = 0; i < J.size(0); i++) {
        auto gs = torch::autograd::grad({r[i]}, cs, {}, true);
        for (at::Tensor & g : gs) if (g.sizes().size() != 1) g = g.view(g.numel());
        J[i].copy_(at::cat(gs));
    }
    return J;
}

at::Tensor compute_Krqc(const at::Tensor & r, const at::Tensor & q, const std::vector<at::Tensor> & cs) {
    int64_t NPars = 0;
    for (const at::Tensor & c : cs) NPars += c.numel();
    at::Tensor K = r.new_empty({r.size(0), q.size(0), NPars});
    for (size_t i = 0; i < K.size(0); i++) {
        auto g = torch::autograd::grad({r[i]}, {q}, {}, true, true);
        for (size_t j = 0; j < K.size(1); j++) {
            auto h = torch::autograd::grad({g[0][j]}, cs, {}, true, false, true);
            for (size_t k = 0; k < cs.size(); k++) {
                if (! h[k].defined()) h[k] = cs[k].new_zeros(cs[k].sizes());
                if (h[k].sizes().size() != 1) h[k] = h[k].view(h[k].numel());
            }
            K[i][j].copy_(at::cat(h));
        }
    }
    return K;
}

at::Tensor merge2T(const at::Tensor & a, const at::Tensor & b) {
    at::Tensor c = a.new_zeros({a.size(0) + b.size(0), a.size(1) + b.size(1)});
    c.slice(0, 0, a.size(0)).slice(1, 0, a.size(1)).copy_(a);
    c.slice(0, a.size(0)).slice(1, a.size(1)).copy_(b);
    return c.transpose(0, 1);
}

at::Tensor merge3(const at::Tensor & a, const at::Tensor & b) {
    at::Tensor c = a.new_zeros({a.size(0) + b.size(0), a.size(1) + b.size(1), a.size(2) + b.size(2)});
    c.slice(0, 0, a.size(0)).slice(1, 0, a.size(1)).slice(2, 0, a.size(2)).copy_(a);
    c.slice(0, a.size(0)).slice(1, a.size(1)).slice(2, a.size(2)).copy_(b);
    return c;
}

void diabatic_DimRed_obnet() {
    std::cout << "Testing Hd computed from *DimRed* and *obnet*...\n\n";

    std::vector<size_t> encoder0_dims = {27, 13, 6, 3},
                        encoder1_dims = {21, 10, 5, 2};
    auto encoder0 = std::make_shared<DimRed::Encoder>(encoder0_dims, true);
    auto encoder1 = std::make_shared<DimRed::Encoder>(encoder1_dims, false);

    Hdnet = std::make_shared<obnet::symat>("DimRed-obnet_Hd.in");

    std::vector<std::string> sapoly_files = {"DimRed-obnet_11.in", "DimRed-obnet_12.in", "DimRed-obnet_22.in"};
    std::vector<size_t> NSASDICs = {3, 2};
    input_generator = std::make_shared<InputGenerator>(Hdnet->NStates(), Hdnet->irreds(), sapoly_files, NSASDICs);

    std::vector<std::string> data = {"min-C1/"};
    abinitio::SAReader reader(data, cart2int);
    std::shared_ptr<abinitio::DataSet<abinitio::RegSAHam>> regset;
    std::shared_ptr<abinitio::DataSet<abinitio::DegSAHam>> degset;
    reader.pretty_print(std::cout);
    std::tie(regset, degset) = reader.read_SAHamSet();

    std::shared_ptr<abinitio::RegSAHam> example = regset->get(0);
    std::vector<at::Tensor> qs = example->qs();
    for (at::Tensor & q : qs) q.set_requires_grad(true);

    std::vector<at::Tensor> rs(2);
    rs[0] = encoder0->forward(qs[0]);
    rs[1] = encoder1->forward(qs[1]);
    // dr / dq
    at::Tensor Jrq0 = compute_Jrq(rs[0], qs[0]),
               Jrq1 = compute_Jrq(rs[1], qs[1]);
    at::Tensor JrqT = merge2T(Jrq0, Jrq1);
    // dr / dc
    auto cs_encoder0 = encoder0->parameters();
    at::Tensor Jrc0 = compute_Jrc(rs[0], cs_encoder0);
    int64_t NPars0 = Jrc0.size(1);
    auto cs_encoder1 = encoder1->parameters();
    at::Tensor Jrc1 = compute_Jrc(rs[1], cs_encoder1);
    int64_t NPars1 = Jrc1.size(1);
    at::Tensor JrcT = merge2T(Jrc0, Jrc1);
    // ddr / dq / dc
    at::Tensor Krqc0 = compute_Krqc(rs[0], qs[0], cs_encoder0),
               Krqc1 = compute_Krqc(rs[1], qs[1], cs_encoder1);
    at::Tensor Krqc = merge3(Krqc0, Krqc1);

    CL::utility::matrix<at::Tensor> ls, JlrTs;
    std::tie(ls, JlrTs) = int2input(rs);
    at::Tensor Hd = Hdnet->forward(ls);

    at::Tensor DqHd = Hderiva::DxHd(Hd, ls, JlrTs, JrqT);
    at::Tensor DqHd_A = DqHd.new_empty(DqHd.sizes());
    for (size_t i = 0; i < Hd.size(0); i++)
    for (size_t j = i; j < Hd.size(1); j++) {
        auto g = torch::autograd::grad({Hd[i][j]}, qs, {}, true, true);
        DqHd_A[i][j] = at::cat(g);
    }
    double difference = 0.0;
    for (size_t i = 0; i < Hdnet->NStates(); i++)
    for (size_t j = i; j < Hdnet->NStates(); j++)
    difference += (DqHd[i][j] - DqHd_A[i][j]).pow(2).sum().item<double>();
    std::cout << "\nd / dx * Hd: " << sqrt(difference) << '\n';

    auto cs_obnet = Hdnet->elements->parameters();
    at::Tensor DcHd = Hderiva::DcHd(Hd, ls, cs_obnet, JlrTs, JrcT);
    std::vector<at::Tensor> parameters = cs_encoder0;
    parameters.insert(parameters.end(), cs_encoder1.begin(), cs_encoder1.end());
    parameters.insert(parameters.end(), cs_obnet.begin(), cs_obnet.end());
    at::Tensor DcHd_A = Hderiva::DcHd(Hd, parameters);
    difference = 0.0;
    for (size_t i = 0; i < Hdnet->NStates(); i++)
    for (size_t j = i; j < Hdnet->NStates(); j++)
    difference += (DcHd[i][j] - DcHd_A[i][j]).pow(2).sum().item<double>();
    std::cout << "\nd / dc * Hd: " << sqrt(difference) << '\n';

    CL::utility::matrix<at::Tensor> Klrs = input_generator->compute_K(rs);
    at::Tensor DcDqHd = Hderiva::DcDxHd(Hd, ls, cs_obnet, JlrTs, Klrs, JrqT, JrcT, Krqc);
    at::Tensor DcDqHd_A = Hderiva::DcDxHd(DqHd_A, parameters);
    difference = 0.0;
    for (size_t i = 0; i < Hdnet->NStates(); i++)
    for (size_t j = i; j < Hdnet->NStates(); j++)
    difference += (DcDqHd[i][j] - DcDqHd_A[i][j]).pow(2).sum().item<double>();
    std::cout << "\nd / dc * d / dx * Hd: " << sqrt(difference) << '\n';
}