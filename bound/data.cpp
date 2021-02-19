#include <abinitio/SAreader.hpp>

#include "global.hpp"

#include "data.hpp"

RegHam::RegHam() {}
RegHam::RegHam(const std::shared_ptr<abinitio::RegSAHam> & ham,
std::tuple<CL::utility::matrix<at::Tensor>, CL::utility::matrix<at::Tensor>> (*q2x)(const std::vector<at::Tensor> &)) {
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
    std::tie(xs_, JTs_) = q2x(qs_);
}
RegHam::~RegHam() {}

CL::utility::matrix<at::Tensor> RegHam::xs() const {return xs_;};
CL::utility::matrix<at::Tensor> RegHam::JTs() const {return JTs_;};





DegHam::DegHam() {}
DegHam::DegHam(const std::shared_ptr<abinitio::DegSAHam> & ham,
std::tuple<CL::utility::matrix<at::Tensor>, CL::utility::matrix<at::Tensor>> (*q2x)(const std::vector<at::Tensor> &)) {
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
    std::tie(xs_, JTs_) = q2x(qs_);
}
DegHam::~DegHam() {}

CL::utility::matrix<at::Tensor> DegHam::xs() const {return xs_;};
CL::utility::matrix<at::Tensor> DegHam::JTs() const {return JTs_;};





std::tuple<std::shared_ptr<abinitio::DataSet<RegHam>>, std::shared_ptr<abinitio::DataSet<DegHam>>>
read_data(const std::vector<std::string> & user_list) {
    abinitio::SAReader reader(user_list, cart2int);
    reader.pretty_print(std::cout);
    // Read the data set in symmetry adapted internal coordinate in standard form
    std::shared_ptr<abinitio::DataSet<abinitio::RegSAHam>> stdregset;
    std::shared_ptr<abinitio::DataSet<abinitio::DegSAHam>> stddegset;
    std::tie(stdregset, stddegset) = reader.read_SAHamSet();
    // Precompute the input layers for each geometry
    std::vector<std::shared_ptr<RegHam>> pregs(stdregset->size_int());
    #pragma omp parallel for
    for (size_t i = 0; i < pregs.size(); i++)
    pregs[i] = std::make_shared<RegHam>(stdregset->get(i), int2input);
    std::vector<std::shared_ptr<DegHam>> pdegs(stddegset->size_int());
    #pragma omp parallel for
    for (size_t i = 0; i < pdegs.size(); i++)
    pdegs[i] = std::make_shared<DegHam>(stddegset->get(i), int2input);
    // Return
    std::shared_ptr<abinitio::DataSet<RegHam>> regset = std::make_shared<abinitio::DataSet<RegHam>>(pregs);
    std::shared_ptr<abinitio::DataSet<DegHam>> degset = std::make_shared<abinitio::DataSet<DegHam>>(pdegs);
    return std::make_tuple(regset, degset);
}