#include <Hderiva/diabatic.hpp>

#include <Hd/kernel.hpp>

namespace Hd {

kernel::kernel() {}
kernel::kernel(
const std::string & format, const std::string & IC, const std::string & SAS,
const std::string & net1, const std::string & checkpoint1,
const std::vector<std::string> & input_layers1,
const std::string & net2, const std::string & checkpoint2,
const std::vector<std::string> & input_layers2) {
    sasicset_ = std::make_shared<SASDIC::SASDICSet>(format, IC, SAS);
    // network 1
    Hdnet1_ = std::make_shared<obnet::symat>(net1);
    torch::load(Hdnet1_->elements, checkpoint1);
    Hdnet1_->freeze();
    Hdnet1_->eval();
    input_generator1_ = std::make_shared<InputGenerator>(Hdnet1_->NStates(), Hdnet1_->irreds(), input_layers1, sasicset_->NSASDICs());
    // network 2
    Hdnet2_ = std::make_shared<obnet::symat>(net2);
    torch::load(Hdnet2_->elements, checkpoint2);
    Hdnet2_->freeze();
    Hdnet2_->eval();
    input_generator2_ = std::make_shared<InputGenerator>(Hdnet2_->NStates(), Hdnet2_->irreds(), input_layers2, sasicset_->NSASDICs());
}
kernel::kernel(const std::vector<std::string> & args) {
    std::string format = args[0],
                IC     = args[1],
                SAS    = args[2];
    sasicset_ = std::make_shared<SASDIC::SASDICSet>(format, IC, SAS);
    // network definitions
    std::vector<std::string> nets = std::vector<std::string>(args.begin() + 3, args.end());
    size_t half = nets.size() / 2;
    std::vector<std::string> net1s = std::vector<std::string>(nets.begin(), nets.begin() + half),
                             net2s = std::vector<std::string>(nets.begin() + half, nets.end()); 
    std::string net1        = net1s[0],
                checkpoint1 = net1s[1];
    std::vector<std::string> input_layers1 = std::vector<std::string>(net1s.begin() + 2, net1s.end());
    std::string net2        = net2s[0],
                checkpoint2 = net2s[1];
    std::vector<std::string> input_layers2 = std::vector<std::string>(net2s.begin() + 2, net2s.end());
    // network 1
    Hdnet1_ = std::make_shared<obnet::symat>(net1);
    torch::load(Hdnet1_->elements, checkpoint1);
    Hdnet1_->freeze();
    Hdnet1_->eval();
    input_generator1_ = std::make_shared<InputGenerator>(Hdnet1_->NStates(), Hdnet1_->irreds(), input_layers1, sasicset_->NSASDICs());
    // network 2
    Hdnet2_ = std::make_shared<obnet::symat>(net2);
    torch::load(Hdnet2_->elements, checkpoint2);
    Hdnet2_->freeze();
    Hdnet2_->eval();
    input_generator2_ = std::make_shared<InputGenerator>(Hdnet2_->NStates(), Hdnet2_->irreds(), input_layers2, sasicset_->NSASDICs());
}
kernel::~kernel() {}

size_t kernel::NStates() const {return Hdnet1_->NStates();}

// given Cartesian coordinate r, return Hd
at::Tensor kernel::operator()(const at::Tensor & r) const {
    if (r.sizes().size() != 1) throw std::invalid_argument(
    "Hd::kernel::operator(): r must be a vector");
    // Cartesian coordinate -> CNPI group symmetry adaptated and scaled internal coordinate
    at::Tensor q = sasicset_->tchem::IC::IntCoordSet::operator()(r);
    std::vector<at::Tensor> qs = (*sasicset_)(q);
    // SASDIC -> input layer
    CL::utility::matrix<at::Tensor> xs1 = (*input_generator1_)(qs),
                                    xs2 = (*input_generator2_)(qs);
    // input layer -> Hd
    return (*Hdnet1_)(xs1) + (*Hdnet2_)(xs2);
}
// given CNPI group symmetry adapted and scaled internal coordinate, return Hd
at::Tensor kernel::operator()(const std::vector<at::Tensor> & qs) const {
    if (qs.size() != sasicset_->NIrreds()) throw std::invalid_argument(
    "Hd::kernel::operator(): qs has wrong number of irreducibles");
    auto NSASDICs = sasicset_->NSASDICs();
    for (size_t i = 0; i < qs.size(); i++)
    if (qs[i].size(0) != NSASDICs[i]) throw std::invalid_argument(
    "Hd::kernel::operator(): qs has wrong number of internal coordinates");
    // SASDIC -> input layer
    CL::utility::matrix<at::Tensor> xs1 = (*input_generator1_)(qs),
                                    xs2 = (*input_generator2_)(qs);
    // input layer -> Hd
    return (*Hdnet1_)(xs1) + (*Hdnet2_)(xs2);
}

// given Cartesian coordinate r, return Hd and ▽Hd
std::tuple<at::Tensor, at::Tensor> kernel::compute_Hd_dHd(const at::Tensor & r) const {
    if (r.sizes().size() != 1) throw std::invalid_argument(
    "Hd::kernel::compute_Hd_dHd: r must be a vector");
    // Cartesian coordinate -> CNPI group symmetry adaptated and scaled internal coordinate
    at::Tensor q, J;
    std::tie(q, J) = sasicset_->compute_IC_J(r);
    q.set_requires_grad(true);
    std::vector<at::Tensor> qs = (*sasicset_)(q);
    std::vector<at::Tensor> Jqrs = std::vector<at::Tensor>(qs.size());
    for (size_t i = 0; i < qs.size(); i++) {
        Jqrs[i] = qs[i].new_empty({qs[i].size(0), q.size(0)});
        for (size_t j = 0; j < qs[i].size(0); j++) {
            std::vector<at::Tensor> g = torch::autograd::grad({qs[i][j]}, {q}, {}, true);
            Jqrs[i][j].copy_(g[0]);
        }
        Jqrs[i] = Jqrs[i].mm(J);
    }
    at::Tensor JqrT = at::cat(Jqrs).transpose(0, 1);
    for (at::Tensor & q : qs) q.detach_();
    // SASDIC -> input layer
    size_t NStates = Hdnet1_->NStates();
    CL::utility::matrix<at::Tensor> xs1(NStates), JxqTs1(NStates),
                                    xs2(NStates), JxqTs2(NStates);
    std::tie(xs1, JxqTs1) = input_generator1_->compute_x_JT(qs);
    std::tie(xs2, JxqTs2) = input_generator2_->compute_x_JT(qs);
    // input layer -> Hd and SASDIC ▽Hd
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = i; j < NStates; j++) {
        xs1[i][j].set_requires_grad(true);
        xs2[i][j].set_requires_grad(true);
    }
    at::Tensor Hd1 = (*Hdnet1_)(xs1),
               Hd2 = (*Hdnet2_)(xs2);
    at::Tensor DqHd = Hderiva::DxHd(Hd1, xs1, JxqTs1) + Hderiva::DxHd(Hd2, xs2, JxqTs2);
    Hd1.detach_();
    Hd2.detach_();
    // SASDIC ▽Hd -> Cartesian coordinate ▽Hd
    at::Tensor DrHd = Hd1.new_empty({Hd1.size(0), Hd1.size(1), r.size(0)});
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = i; j < NStates; j++)
    DrHd[i][j] = JqrT.mv(DqHd[i][j]);
    return std::make_tuple(Hd1 + Hd2, DrHd);
}
// given CNPI group symmetry adapted and scaled internal coordinate, return Hd and ▽Hd
std::tuple<at::Tensor, at::Tensor> kernel::compute_Hd_dHd(const std::vector<at::Tensor> & qs) const {
    // SASDIC -> input layer
    size_t NStates = Hdnet1_->NStates();
    CL::utility::matrix<at::Tensor> xs1(NStates), JxqTs1(NStates),
                                    xs2(NStates), JxqTs2(NStates);
    std::tie(xs1, JxqTs1) = input_generator1_->compute_x_JT(qs);
    std::tie(xs2, JxqTs2) = input_generator2_->compute_x_JT(qs);
    // input layer -> Hd and SASDIC ▽Hd
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = i; j < NStates; j++) {
        xs1[i][j].set_requires_grad(true);
        xs2[i][j].set_requires_grad(true);
    }
    at::Tensor Hd1 = (*Hdnet1_)(xs1),
               Hd2 = (*Hdnet2_)(xs2);
    at::Tensor DqHd = Hderiva::DxHd(Hd1, xs1, JxqTs1) + Hderiva::DxHd(Hd2, xs2, JxqTs2);
    Hd1.detach_();
    Hd2.detach_();
    return std::make_tuple(Hd1 + Hd2, DqHd);
}

// output hidden layer values before activation to `os`
void kernel::diagnostic(const at::Tensor & r, std::ostream & os) {
    if (r.sizes().size() != 1) throw std::invalid_argument(
    "Hd::kernel::operator(): r must be a vector");
    // Cartesian coordinate -> CNPI group symmetry adaptated and scaled internal coordinate
    at::Tensor q = sasicset_->tchem::IC::IntCoordSet::operator()(r);
    std::vector<at::Tensor> qs = (*sasicset_)(q);
    // SASDIC -> input layer
    CL::utility::matrix<at::Tensor> xs1 = (*input_generator1_)(qs),
                                    xs2 = (*input_generator2_)(qs);
    // input layer -> Hd
    Hdnet1_->diagnostic(xs1, os);
    Hdnet2_->diagnostic(xs2, os);
}

} // namespace Hd