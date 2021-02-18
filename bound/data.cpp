#include <abinitio/SAreader.hpp>

#include "global.hpp"

#include "data.hpp"

RegHam::RegHam() {}
RegHam::RegHam(const std::shared_ptr<abinitio::RegSAHam> & ham,
CL::utility::matrix<at::Tensor> (*q2x)(const std::vector<at::Tensor> &)) {
    CNPI2point_ = ham->CNPI2point();
    qs_         = ham->qs        ();
    Js_         = ham->Js        ();
    point2CNPI_ = ham->point2CNPI();
    S_          = ham->S         ();
    Ss_         = ham->Ss        ();
    sqrtSs_     = ham->sqrtSs    ();
    weight_     = ham->weight    ();
    energy_     = ham->energy    ();
    dH_         = ham->dH        ();
    irreds_     = ham->irreds    ();
    // Construct `xs_` and `JTs_`
    for (at::Tensor & q : qs_) q.set_requires_grad(true);
    xs_ = q2x(qs_);
    size_t NStates = xs_.size();
    int64_t intdim = 0;
    for (const at::Tensor & q : qs_) intdim += q.size(0);
    JTs_.resize(NStates);
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = i; j < NStates; j++) {
        at::Tensor & x = xs_[i][j], & JT = JTs_[i][j];
        JT = x.new_empty({intdim, x.size(0)});
        std::vector<at::Tensor> CNPIviews = this->split2CNPI(JT);
        for (at::Tensor & JTT : CNPIviews) JTT.transpose_(0, 1);
        for (size_t row = 0; row < x.size(0); row++) {
            torch::autograd::variable_list g = torch::autograd::grad({x[row]}, qs_, {}, true);
            for (size_t irred = 0; irred < qs_.size(); irred++)
            CNPIviews[irred][row] = g[irred]; 
        }
        for (at::Tensor & JTTT : CNPIviews) JTTT.transpose_(0, 1);
    }
    // Free autograd graph
    for (at::Tensor & q : qs_) {
        q.detach_();
        q.set_requires_grad(false);
    }
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = i; j < NStates; j++)
    xs_[i][j].detach_();
}
RegHam::~RegHam() {}

CL::utility::matrix<at::Tensor> RegHam::xs() const {return xs_;};
CL::utility::matrix<at::Tensor> RegHam::JTs() const {return JTs_;};





DegHam::DegHam() {}
DegHam::DegHam(const std::shared_ptr<abinitio::DegSAHam> & ham,
CL::utility::matrix<at::Tensor> (*q2x)(const std::vector<at::Tensor> &)) {
    CNPI2point_ = ham->CNPI2point();
    qs_         = ham->qs        ();
    Js_         = ham->Js        ();
    point2CNPI_ = ham->point2CNPI();
    S_          = ham->S         ();
    Ss_         = ham->Ss        ();
    sqrtSs_     = ham->sqrtSs    ();
    weight_     = ham->weight    ();
    energy_     = ham->energy    ();
    dH_         = ham->dH        ();
    irreds_     = ham->irreds    ();
    H_          = ham->H         ();
    // Construct `xs_` and `JTs_`
    for (at::Tensor & q : qs_) q.set_requires_grad(true);
    xs_ = q2x(qs_);
    size_t NStates = xs_.size();
    int64_t intdim = 0;
    for (const at::Tensor & q : qs_) intdim += q.size(0);
    JTs_.resize(NStates);
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = i; j < NStates; j++) {
        at::Tensor & x = xs_[i][j], & JT = JTs_[i][j];
        JT = x.new_empty({intdim, x.size(0)});
        std::vector<at::Tensor> CNPIviews = this->split2CNPI(JT);
        for (at::Tensor & JTT : CNPIviews) JTT.transpose_(0, 1);
        for (size_t row = 0; row < x.size(0); row++) {
            torch::autograd::variable_list g = torch::autograd::grad({x[row]}, qs_, {}, true);
            for (size_t irred = 0; irred < qs_.size(); irred++)
            CNPIviews[irred][row] = g[irred]; 
        }
        for (at::Tensor & JTTT : CNPIviews) JTTT.transpose_(0, 1);
    }
    // Free autograd graph
    for (at::Tensor & q : qs_) {
        q.detach_();
        q.set_requires_grad(false);
    }
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = i; j < NStates; j++)
    xs_[i][j].detach_();
}
DegHam::~DegHam() {}

CL::utility::matrix<at::Tensor> DegHam::xs() const {return xs_;};
CL::utility::matrix<at::Tensor> DegHam::JTs() const {return JTs_;};





std::tuple<std::shared_ptr<abinitio::DataSet<RegHam>>, std::shared_ptr<abinitio::DataSet<DegHam>>>
read_data(const std::vector<std::string> & user_list) {
    abinitio::SAReader reader(user_list, cart2int);
    reader.pretty_print(std::cout);
    // Read the data set in symmetry adapted internal coordinate
    std::shared_ptr<abinitio::DataSet<abinitio::RegSAHam>> stdregset;
    std::shared_ptr<abinitio::DataSet<abinitio::DegSAHam>> stddegset;
    std::tie(stdregset, stddegset) = reader.read_SAHamSet();
    // Precompute the input layers for each geometry
    std::vector<std::shared_ptr<RegHam>> pregs(stdregset->size_int());
    for (size_t i = 0; i < pregs.size(); i++)
    pregs[i] = std::make_shared<RegHam>(stdregset->get(i), int2input);
    std::vector<std::shared_ptr<DegHam>> pdegs(stddegset->size_int());
    for (size_t i = 0; i < pdegs.size(); i++)
    pdegs[i] = std::make_shared<DegHam>(stddegset->get(i), int2input);
    // Return
    std::shared_ptr<abinitio::DataSet<RegHam>> regset = std::make_shared<abinitio::DataSet<RegHam>>(pregs);
    std::shared_ptr<abinitio::DataSet<DegHam>> degset = std::make_shared<abinitio::DataSet<DegHam>>(pdegs);
    return std::make_tuple(regset, degset);
}