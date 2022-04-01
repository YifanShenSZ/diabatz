#include <Hderiva/diabatic.hpp>

#include <Hd/kernel.hpp>

namespace Hd {

kernel::kernel() {}
kernel::kernel(
const std::string & format1, const std::string & IC1, const std::string & SAS1,
const std::string & net1, const std::string & checkpoint1,
const std::vector<std::string> & input_layers1,
const std::string & format2, const std::string & IC2, const std::string & SAS2,
const std::string & net2, const std::string & checkpoint2,
const std::vector<std::string> & input_layers2) {
    // network 1
    sasicset1_ = std::make_shared<SASDIC::SASDICSet>(format1, IC1, SAS1);
    Hdnet1_ = std::make_shared<obnet::symat>(net1);
    torch::load(Hdnet1_->elements, checkpoint1);
    Hdnet1_->freeze();
    Hdnet1_->eval();
    input_generator1_ = std::make_shared<InputGenerator>(Hdnet1_->NStates(), Hdnet1_->irreds(), input_layers1, sasicset1_->NSASDICs());
    // network 2
    sasicset2_ = std::make_shared<SASDIC::SASDICSet>(format2, IC2, SAS2);
    Hdnet2_ = std::make_shared<obnet::symat>(net2);
    torch::load(Hdnet2_->elements, checkpoint2);
    Hdnet2_->freeze();
    Hdnet2_->eval();
    input_generator2_ = std::make_shared<InputGenerator>(Hdnet2_->NStates(), Hdnet2_->irreds(), input_layers2, sasicset2_->NSASDICs());
}
kernel::kernel(const std::vector<std::string> & args) {
    size_t half = args.size() / 2;
    std::vector<std::string> arg1s = std::vector<std::string>(args.begin(), args.begin() + half),
                             arg2s = std::vector<std::string>(args.begin() + half, args.end());
    std::string format1 = arg1s[0], IC1 = arg1s[1], SAS1 = arg1s[2], net1 = arg1s[3], chk1 = arg1s[4],
                format2 = arg2s[0], IC2 = arg2s[1], SAS2 = arg2s[2], net2 = arg2s[3], chk2 = arg2s[4];
    std::vector<std::string> input_layers1 = std::vector<std::string>(arg1s.begin() + 5, arg1s.end()),
                             input_layers2 = std::vector<std::string>(arg2s.begin() + 5, arg2s.end());
    // network 1
    sasicset1_ = std::make_shared<SASDIC::SASDICSet>(format1, IC1, SAS1);
    Hdnet1_ = std::make_shared<obnet::symat>(net1);
    torch::load(Hdnet1_->elements, chk1);
    Hdnet1_->freeze();
    Hdnet1_->eval();
    input_generator1_ = std::make_shared<InputGenerator>(Hdnet1_->NStates(), Hdnet1_->irreds(), input_layers1, sasicset1_->NSASDICs());
    // network 2
    sasicset2_ = std::make_shared<SASDIC::SASDICSet>(format2, IC2, SAS2);
    Hdnet2_ = std::make_shared<obnet::symat>(net2);
    torch::load(Hdnet2_->elements, chk2);
    Hdnet2_->freeze();
    Hdnet2_->eval();
    input_generator2_ = std::make_shared<InputGenerator>(Hdnet2_->NStates(), Hdnet2_->irreds(), input_layers2, sasicset2_->NSASDICs());
}
kernel::~kernel() {}

size_t kernel::NStates() const {return Hdnet1_->NStates();}

// given Cartesian coordinate r, return Hd
at::Tensor kernel::operator()(const at::Tensor & r) const {
    if (r.sizes().size() != 1) throw std::invalid_argument(
    "Hd::kernel::operator(): r must be a vector");
    // Cartesian coordinate -> CNPI group symmetry adaptated and scaled internal coordinate
    at::Tensor q1 = sasicset1_->tchem::IC::IntCoordSet::operator()(r),
               q2 = sasicset2_->tchem::IC::IntCoordSet::operator()(r);
    std::vector<at::Tensor> q1s = (*sasicset1_)(q1),
                            q2s = (*sasicset2_)(q2);
    // SASDIC -> input layer
    CL::utility::matrix<at::Tensor> x1s = (*input_generator1_)(q1s),
                                    x2s = (*input_generator2_)(q2s);
    // input layer -> Hd
    return (*Hdnet1_)(x1s) + (*Hdnet2_)(x2s);
}

// given Cartesian coordinate r, return Hd and ▽Hd
std::tuple<at::Tensor, at::Tensor> kernel::compute_Hd_dHd(const at::Tensor & r) const {
    if (r.sizes().size() != 1) throw std::invalid_argument(
    "Hd::kernel::compute_Hd_dHd: r must be a vector");
    // Cartesian coordinate -> CNPI group symmetry adaptated and scaled internal coordinate
    at::Tensor q1, J1, q2, J2;
    std::tie(q1, J1) = sasicset1_->compute_IC_J(r);
    std::tie(q2, J2) = sasicset2_->compute_IC_J(r);
    q1.set_requires_grad(true);
    q2.set_requires_grad(true);
    std::vector<at::Tensor> q1s = (*sasicset1_)(q1),
                            q2s = (*sasicset2_)(q2);
    std::vector<at::Tensor> Jq1rs = std::vector<at::Tensor>(q1s.size()),
                            Jq2rs = std::vector<at::Tensor>(q2s.size());
    for (size_t i = 0; i < q1s.size(); i++) {
        Jq1rs[i] = q1s[i].new_empty({q1s[i].size(0), q1.size(0)});
        for (size_t j = 0; j < q1s[i].size(0); j++) {
            std::vector<at::Tensor> g = torch::autograd::grad({q1s[i][j]}, {q1}, {}, true);
            Jq1rs[i][j].copy_(g[0]);
        }
        Jq1rs[i] = Jq1rs[i].mm(J1);
    }
    for (size_t i = 0; i < q2s.size(); i++) {
        Jq2rs[i] = q2s[i].new_empty({q2s[i].size(0), q2.size(0)});
        for (size_t j = 0; j < q2s[i].size(0); j++) {
            std::vector<at::Tensor> g = torch::autograd::grad({q2s[i][j]}, {q2}, {}, true);
            Jq2rs[i][j].copy_(g[0]);
        }
        Jq2rs[i] = Jq2rs[i].mm(J2);
    }
    at::Tensor Jq1rT = at::cat(Jq1rs).transpose(0, 1);
    at::Tensor Jq2rT = at::cat(Jq2rs).transpose(0, 1);
    for (at::Tensor & q : q1s) q.detach_();
    for (at::Tensor & q : q2s) q.detach_();
    // SASDIC -> input layer
    size_t NStates = Hdnet1_->NStates();
    CL::utility::matrix<at::Tensor> x1s(NStates), JxqT1s(NStates),
                                    x2s(NStates), JxqT2s(NStates);
    std::tie(x1s, JxqT1s) = input_generator1_->compute_x_JT(q1s);
    std::tie(x2s, JxqT2s) = input_generator2_->compute_x_JT(q2s);
    // input layer -> Hd and SASDIC ▽Hd
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = i; j < NStates; j++) {
        x1s[i][j].set_requires_grad(true);
        x2s[i][j].set_requires_grad(true);
    }
    at::Tensor Hd1 = (*Hdnet1_)(x1s),
               Hd2 = (*Hdnet2_)(x2s);
    at::Tensor DqHd1 = Hderiva::DxHd(Hd1, x1s, JxqT1s),
               DqHd2 = Hderiva::DxHd(Hd2, x2s, JxqT2s);
    Hd1.detach_();
    Hd2.detach_();
    // SASDIC ▽Hd -> Cartesian coordinate ▽Hd
    at::Tensor DrHd = Hd1.new_empty({Hd1.size(0), Hd1.size(1), r.size(0)});
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = i; j < NStates; j++)
    DrHd[i][j] = Jq1rT.mv(DqHd1[i][j]) + Jq2rT.mv(DqHd2[i][j]);
    return std::make_tuple(Hd1 + Hd2, DrHd);
}

// output hidden layer values before activation to `os`
void kernel::diagnostic(const at::Tensor & r, std::ostream & os) {
    if (r.sizes().size() != 1) throw std::invalid_argument(
    "Hd::kernel::operator(): r must be a vector");
    // Cartesian coordinate -> CNPI group symmetry adaptated and scaled internal coordinate
    at::Tensor q1 = sasicset1_->tchem::IC::IntCoordSet::operator()(r),
               q2 = sasicset2_->tchem::IC::IntCoordSet::operator()(r);
    std::vector<at::Tensor> q1s = (*sasicset1_)(q1),
                            q2s = (*sasicset2_)(q2);
    // SASDIC -> input layer
    CL::utility::matrix<at::Tensor> x1s = (*input_generator1_)(q1s),
                                    x2s = (*input_generator2_)(q2s);
    // input layer -> Hd
    Hdnet1_->diagnostic(x1s, os);
    Hdnet2_->diagnostic(x2s, os);
}

} // namespace Hd