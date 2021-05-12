#include <abinitio/SAreader.hpp>

#include "../include/global.hpp"

#include "../include/data.hpp"

RegHam::RegHam() {}
RegHam::RegHam(const std::shared_ptr<abinitio::RegSAHam> & ham,
std::tuple<CL::utility::matrix<at::Tensor>, CL::utility::matrix<at::Tensor>> (*q2x)(const std::vector<at::Tensor> &))
: abinitio::RegSAHam(*ham) {
    std::tie(xs_, JxqTs_) = q2x(qs_);
}
RegHam::~RegHam() {}

const CL::utility::matrix<at::Tensor> & RegHam::xs() const {return xs_;};
const CL::utility::matrix<at::Tensor> & RegHam::JxqTs() const {return JxqTs_;};





DegHam::DegHam() {}
DegHam::DegHam(const std::shared_ptr<abinitio::DegSAHam> & ham,
std::tuple<CL::utility::matrix<at::Tensor>, CL::utility::matrix<at::Tensor>> (*q2x)(const std::vector<at::Tensor> &))
: abinitio::DegSAHam(*ham) {
    std::tie(xs_, JxqTs_) = q2x(qs_);
}
DegHam::~DegHam() {}

const CL::utility::matrix<at::Tensor> & DegHam::xs() const {return xs_;};
const CL::utility::matrix<at::Tensor> & DegHam::JxqTs() const {return JxqTs_;};





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