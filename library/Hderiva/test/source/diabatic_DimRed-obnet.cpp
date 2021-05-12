#include <abinitio/SAreader.hpp>

#include <DimRed/encoder.hpp>
#include <DimRed/decoder.hpp>

#include <Hderiva/diabatic.hpp>

#include "../include/global.hpp"

void diabatic_DimRed_obnet() {
    std::cout << "Testing Hd computed from *DimRed* and *obnet*...\n\n";

    std::vector<size_t> encoder0_dims = {27, 13, 6, 3},
                        decoder0_dims = {3, 6, 13, 27};
    auto encoder0 = std::make_shared<DimRed::Encoder>(encoder0_dims, true);
    auto decoder0 = std::make_shared<DimRed::Decoder>(decoder0_dims, true);
    std::vector<size_t> encoder1_dims = {21, 10, 5, 2},
                        decoder1_dims = {2, 5, 10, 21};
    auto encoder1 = std::make_shared<DimRed::Encoder>(encoder1_dims, false);
    auto decoder1 = std::make_shared<DimRed::Decoder>(decoder1_dims, false);

    Hdnet = std::make_shared<obnet::symat>("DimRed-obnet_Hd.in");
    Hdnet->to(torch::kFloat64);

    std::vector<std::string> sapoly_files = {"DimRed-obnet_11.in", "DimRed-obnet_12.in", "DimRed-obnet_22.in"};
    std::vector<size_t> NSASICs = {3, 2};
    input_generator = std::make_shared<InputGenerator>(Hdnet->NStates(), Hdnet->irreds(), sapoly_files, NSASICs);

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
    at::Tensor Jrq0 = rs[0].new_empty({rs[0].size(0), qs[0].size(0)});
    for (size_t i = 0; i < Jrq0.size(0); i++) {
        auto g = torch::autograd::grad({rs[0][i]}, {qs[0]}, {}, true);
        Jrq0[i].copy_(g[0]);
    }
    at::Tensor Jrq1 = rs[1].new_empty({rs[1].size(0), qs[1].size(0)});
    for (size_t i = 0; i < Jrq1.size(0); i++) {
        auto g = torch::autograd::grad({rs[1][i]}, {qs[1]}, {}, true);
        Jrq1[i].copy_(g[0]);
    }
    at::Tensor Jrq = Jrq0.new_zeros({Jrq0.size(0) + Jrq1.size(0), Jrq0.size(1) + Jrq1.size(1)});
    Jrq.slice(0, 0, Jrq0.size(0)).slice(1, 0, Jrq0.size(1)).copy_(Jrq0);
    Jrq.slice(0, Jrq0.size(0)).slice(1, Jrq0.size(1)).copy_(Jrq1);
    at::Tensor JrqT = Jrq.transpose(0, 1);
    // dr / dc
    auto cs_encoder0 = encoder0->parameters();
    int64_t NPars0 = 0;
    for (const at::Tensor & c : cs_encoder0) NPars0 += c.numel();
    at::Tensor Jrc0 = rs[0].new_empty({rs[0].size(0), NPars0});
    for (size_t i = 0; i < Jrc0.size(0); i++) {
        auto gs = torch::autograd::grad({rs[0][i]}, cs_encoder0, {}, true);
        for (at::Tensor & g : gs) if (g.sizes().size() != 1) g = g.view(g.numel());
        Jrc0[i].copy_(at::cat(gs));
    }
    auto cs_encoder1 = encoder1->parameters();
    int64_t NPars1 = 0;
    for (const at::Tensor & c : cs_encoder1) NPars1 += c.numel();
    at::Tensor Jrc1 = rs[0].new_empty({rs[1].size(0), NPars1});
    for (size_t i = 0; i < Jrc1.size(0); i++) {
        auto gs = torch::autograd::grad({rs[1][i]}, cs_encoder1, {}, true);
        for (at::Tensor & g : gs) if (g.sizes().size() != 1) g = g.view(g.numel());
        Jrc1[i].copy_(at::cat(gs));
    }
    at::Tensor Jrc = Jrc0.new_zeros({Jrc0.size(0) + Jrc1.size(0), Jrc0.size(1) + Jrc1.size(1)});
    Jrc.slice(0, 0, Jrc0.size(0)).slice(1, 0, Jrc0.size(1)).copy_(Jrc0);
    Jrc.slice(0, Jrc0.size(0)).slice(1, Jrc0.size(1)).copy_(Jrc1);
    at::Tensor JrcT = Jrc.transpose(0, 1);
    // ddr / dq / dc
    at::Tensor Krqc0 = rs[0].new_empty({rs[0].size(0), qs[0].size(0), NPars0});
    for (size_t i = 0; i < Krqc0.size(0); i++) {
        auto g = torch::autograd::grad({rs[0][i]}, {qs[0]}, {}, true, true);
        for (size_t j = 0; j < Krqc0.size(1); j++) {
            auto h = torch::autograd::grad({g[0][j]}, cs_encoder0, {}, true, false, true);
            for (size_t k = 0; k < cs_encoder0.size(); k++) {
                if (! h[k].defined()) h[k] = cs_encoder0[k].new_zeros(cs_encoder0[k].sizes());
                if (h[k].sizes().size() != 1) h[k] = h[k].view(h[k].numel());
            }
            Krqc0[i][j].copy_(at::cat(h));
        }
    }
    at::Tensor Krqc1 = rs[1].new_empty({rs[1].size(0), qs[1].size(0), NPars1});
    for (size_t i = 0; i < Krqc1.size(0); i++) {
        auto g = torch::autograd::grad({rs[1][i]}, {qs[1]}, {}, true, true);
        for (size_t j = 0; j < Krqc1.size(1); j++) {
            auto h = torch::autograd::grad({g[0][j]}, cs_encoder1, {}, true);
            for (at::Tensor & grad : h) if (grad.sizes().size() != 1) grad = grad.view(grad.numel());
            Krqc1[i][j].copy_(at::cat(h));
        }
    }
    at::Tensor Krqc = Krqc0.new_zeros({Krqc0.size(0) + Krqc1.size(0), Krqc0.size(1) + Krqc1.size(1), Krqc0.size(2) + Krqc1.size(2)});
    Krqc.slice(0, 0, Krqc0.size(0)).slice(1, 0, Krqc0.size(1)).slice(2, 0, Krqc0.size(2)).copy_(Krqc0);
    Krqc.slice(0, Krqc0.size(0)).slice(1, Krqc0.size(1)).slice(2, Krqc0.size(2)).copy_(Krqc1);
    at::Tensor KrqcT = Krqc.transpose(0, 1);

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

    at::Tensor DcDxHd = Hderiva::DcDxHd(Hd, ls, cs_obnet, JlrTs, JrqT, JrcT, KrqcT);
    at::Tensor DcDxHd_A = Hderiva::DcDxHd(DqHd_A, parameters);
    difference = 0.0;
    for (size_t i = 0; i < Hdnet->NStates(); i++)
    for (size_t j = i; j < Hdnet->NStates(); j++)
    difference += (DcDxHd[i][j] - DcDxHd_A[i][j]).pow(2).sum().item<double>();
    std::cout << "\nd / dc * d / dx * Hd: " << sqrt(difference) << '\n';
    std::cout << "This difference may come from the long backward graph again\n"
              << "The root mean square deviations of each element are:\n";
    for (size_t i = 0; i < Hdnet->NStates(); i++)
    for (size_t j = i; j < Hdnet->NStates(); j++)
    std::cout << "Hd " << i << ' ' << j << '\n'
              << "c from encoder of irreducible 1: "
              << sqrt(
                     (DcDxHd[i][j] - DcDxHd_A[i][j]).slice(1, 0, NPars0).pow(2).sum().item<double>()
                     / DcDxHd[i][j].slice(0, 0, 27).slice(1, 0, NPars0).numel()
                 ) << '\n'
              << "c from encoder of irreducible 2: "
              << sqrt(
                     (DcDxHd[i][j] - DcDxHd_A[i][j]).slice(1, NPars0, NPars0 + NPars1).pow(2).sum().item<double>()
                     / DcDxHd[i][j].slice(0, 27).slice(1, NPars0, NPars0 + NPars1).numel()
                 ) << '\n'
              << "c from obnet works fine: "
              << sqrt(
                     (DcDxHd[i][j] - DcDxHd_A[i][j]).slice(1, NPars0 + NPars1).pow(2).sum().item<double>()
                     / DcDxHd[i][j].slice(1, NPars0 + NPars1).numel()
                 ) << '\n';
    std::cout << "I checked Hd00 and found that for c from encoder of irreducible 1,\n"
              << "The matrix of first layer fluctuates around 1e-4, but the bias fluctuates around 1e-3\n"
              << "The latter layers inherit that 1e-3 fluctuation\n";
}