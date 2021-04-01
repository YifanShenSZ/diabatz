#include <Hderiva/diabatic.hpp>

#include <Hd/kernel.hpp>

namespace Hd {

kernel::kernel() {}
kernel::kernel(
const std::string & format, const std::string & IC, const std::string & SAS,
const std::string & net, const std::string & checkpoint,
const std::vector<std::string> & input_layers) {
    sasicset_ = std::make_shared<tchem::IC::SASICSet>(format, IC, SAS);
    Hdnet_ = std::make_shared<obnet::symat>(net);
    torch::load(Hdnet_->elements, checkpoint);
    Hdnet_->freeze();
    Hdnet_->eval();
    input_generator_ = std::make_shared<InputGenerator>(Hdnet_->NStates(), Hdnet_->irreds(), input_layers, sasicset_->NSASICs());
}
kernel::kernel(const std::vector<std::string> & args) : kernel(
args[0], args[1], args[2],
args[3], args[4],
std::vector<std::string>(args.begin() + 5, args.end())) {}
kernel::~kernel() {}

const std::shared_ptr<obnet::symat> & kernel::Hdnet() const {return Hdnet_;}
const std::shared_ptr<InputGenerator> & kernel::input_generator() const {return input_generator_;}

size_t kernel::NStates() const {return Hdnet_->NStates();}

// Given Cartesian coordinate r, return Hd
at::Tensor kernel::operator()(const at::Tensor & r) const {
    if (r.sizes().size() != 1) throw std::invalid_argument(
    "Hd::kernel::operator(): r must be a vector");
    // Cartesian coordinate -> CNPI group symmetry adaptated and scaled internal coordinate
    at::Tensor q = sasicset_->tchem::IC::IntCoordSet::operator()(r);
    std::vector<at::Tensor> qs = (*sasicset_)(q);
    // SASIC -> input layer
    CL::utility::matrix<at::Tensor> xs = (*input_generator_)(qs);
    // input layer -> Hd
    return (*Hdnet_)(xs);
}
// Given CNPI group symmetry adapted and scaled internal coordinate, return Hd
at::Tensor kernel::operator()(const std::vector<at::Tensor> & qs) const {
    if (qs.size() != sasicset_->NIrreds()) throw std::invalid_argument(
    "Hd::kernel::operator(): qs has wrong number of irreducibles");
    auto NSASICs = sasicset_->NSASICs();
    for (size_t i = 0; i < qs.size(); i++)
    if (qs[i].size(0) != NSASICs[i]) throw std::invalid_argument(
    "Hd::kernel::operator(): qs has wrong number of internal coordinates");
    CL::utility::matrix<at::Tensor> xs = (*input_generator_)(qs);
    // input layer -> Hd
    return (*Hdnet_)(xs);
}

// Given Cartesian coordinate r, return Hd and ▽Hd
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
    // SASIC -> input layer
    size_t NStates = Hdnet_->NStates();
    CL::utility::matrix<at::Tensor> xs(NStates), JxqTs(NStates);
    std::tie(xs, JxqTs) = input_generator_->compute_x_JT(qs);
    // input layer -> Hd and SASIC ▽Hd
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = i; j < NStates; j++)
    xs[i][j].set_requires_grad(true);
    at::Tensor Hd = (*Hdnet_)(xs);
    at::Tensor DqHd = Hderiva::DxHd(Hd, xs, JxqTs);
    Hd.detach_();
    // SASIC ▽Hd -> Cartesian coordinate ▽Hd
    at::Tensor DrHd = Hd.new_empty({Hd.size(0), Hd.size(1), r.size(0)});
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = i; j < NStates; j++)
    DrHd[i][j] = JqrT.mv(DqHd[i][j]);
    return std::make_tuple(Hd, DrHd);
}
// Given CNPI group symmetry adapted and scaled internal coordinate, return Hd and ▽Hd
std::tuple<at::Tensor, at::Tensor> kernel::compute_Hd_dHd(const std::vector<at::Tensor> & qs) const {
    // SASIC -> input layer
    size_t NStates = Hdnet_->NStates();
    CL::utility::matrix<at::Tensor> xs(NStates), JxqTs(NStates);
    std::tie(xs, JxqTs) = input_generator_->compute_x_JT(qs);
    // input layer -> Hd and SASIC ▽Hd
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = i; j < NStates; j++)
    xs[i][j].set_requires_grad(true);
    at::Tensor Hd = (*Hdnet_)(xs);
    at::Tensor DqHd = Hderiva::DxHd(Hd, xs, JxqTs);
    Hd.detach_();
    return std::make_tuple(Hd, DqHd);
}

} // namespace Hd