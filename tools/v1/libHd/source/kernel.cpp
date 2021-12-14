#include <Hderiva/diabatic.hpp>

#include <Hd/kernel.hpp>

namespace Hd {

kernel::kernel() {}
kernel::kernel(
const std::string & BLformat, const std::string & BLIC, const std::string & BLparameters,
const std::string & format, const std::string & IC, const std::string & SAS,
const std::string & net, const std::string & checkpoint,
const std::vector<std::string> & input_layers) {
    // 1/r repulsion
    blset_ = std::make_shared<tchem::IC::IntCoordSet>(BLformat, BLIC);
    as_ = CL::utility::read_vector(BLparameters);
    // Hd network
    sasicset_ = std::make_shared<SASDIC::SASDICSet>(format, IC, SAS);
    Hdnet_ = std::make_shared<obnet::symat>(net);
    torch::load(Hdnet_->elements, checkpoint);
    Hdnet_->freeze();
    Hdnet_->eval();
    input_generator_ = std::make_shared<InputGenerator>(Hdnet_->NStates(), Hdnet_->irreds(), input_layers, sasicset_->NSASDICs());
}
kernel::kernel(const std::vector<std::string> & args) : kernel(
args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
std::vector<std::string>(args.begin() + 8, args.end())) {}
kernel::~kernel() {}

const std::shared_ptr<obnet::symat> & kernel::Hdnet() const {return Hdnet_;}
const std::shared_ptr<InputGenerator> & kernel::input_generator() const {return input_generator_;}

size_t kernel::NStates() const {return Hdnet_->NStates();}

// given Cartesian coordinate r, return Hd
at::Tensor kernel::operator()(const at::Tensor & r) const {
    if (r.sizes().size() != 1) throw std::invalid_argument(
    "Hd::kernel::operator(): r must be a vector");
    // 1/r repulsion
    at::Tensor BLs = (*blset_)(r);
    at::Tensor repulsion = as_[0] / BLs[0];
    for (size_t i = 1; i < as_.size(); i++) repulsion += as_[i] / BLs[i];
    // Cartesian coordinate -> CNPI group symmetry adaptated and scaled internal coordinate
    at::Tensor q = sasicset_->tchem::IC::IntCoordSet::operator()(r);
    std::vector<at::Tensor> qs = (*sasicset_)(q);
    // SASDIC -> input layer
    CL::utility::matrix<at::Tensor> xs = (*input_generator_)(qs);
    // input layer -> Hd
    at::Tensor Hd = (*Hdnet_)(xs);
    // combine 1/r repulsion and Hd network
    for (size_t i = 0; i < Hd.size(0); i++) Hd[i][i] += repulsion;
    return Hd;
}

// given Cartesian coordinate r, return Hd and ▽Hd
std::tuple<at::Tensor, at::Tensor> kernel::compute_Hd_dHd(const at::Tensor & r) const {
    if (r.sizes().size() != 1) throw std::invalid_argument(
    "Hd::kernel::compute_Hd_dHd: r must be a vector");
    // 1/r repulsion
    at::Tensor BLs = (*blset_)(r);
    at::Tensor repulsion = as_[0] / BLs[0];
    at::Tensor intrepulse = BLs.new_empty(BLs.sizes());
    intrepulse[0] = -repulsion / BLs[0];
    for (size_t i = 1; i < as_.size(); i++) {
        at::Tensor temp = as_[i] / BLs[i];
        repulsion += temp;
        intrepulse[i] = -temp / BLs[i];
    }
    at::Tensor cartrepulse = blset_->gradient_int2cart(r, intrepulse);
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
    size_t NStates = Hdnet_->NStates();
    CL::utility::matrix<at::Tensor> xs(NStates), JxqTs(NStates);
    std::tie(xs, JxqTs) = input_generator_->compute_x_JT(qs);
    // input layer -> Hd and SASDIC ▽Hd
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = i; j < NStates; j++)
    xs[i][j].set_requires_grad(true);
    at::Tensor Hd = (*Hdnet_)(xs);
    at::Tensor DqHd = Hderiva::DxHd(Hd, xs, JxqTs);
    Hd.detach_();
    // SASDIC ▽Hd -> Cartesian coordinate ▽Hd
    at::Tensor DrHd = Hd.new_empty({Hd.size(0), Hd.size(1), r.size(0)});
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = i; j < NStates; j++)
    DrHd[i][j] = JqrT.mv(DqHd[i][j]);
    // combine 1/r repulsion and Hd network
    for (size_t i = 0; i < Hd.size(0); i++) {
          Hd[i][i] += repulsion;
        DrHd[i][i] += cartrepulse;
    }
    return std::make_tuple(Hd, DrHd);
}

} // namespace Hd