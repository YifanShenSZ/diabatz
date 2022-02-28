#include "../include/data_classes.hpp"

Energy::Energy() {}
Energy::Energy(const std::shared_ptr<abinitio::SAEnergy> & ham,
std::tuple<CL::utility::matrix<at::Tensor>, CL::utility::matrix<at::Tensor>> (*q2x)(const std::vector<at::Tensor> &))
: abinitio::SAEnergy(*ham) {
    std::tie(xs_, JxrTs_) = q2x(qs_);
    for (size_t i = 0; i < JxrTs_.size(0); i++)
    for (size_t j = i; j < JxrTs_.size(1); j++)
    JxrTs_[i][j] = JqrT_.mm(JxrTs_[i][j]);
}
Energy::~Energy() {}

const CL::utility::matrix<at::Tensor> & Energy::xs() const {return xs_;};
const CL::utility::matrix<at::Tensor> & Energy::JxrTs() const {return JxrTs_;};





RegHam::RegHam() {}
RegHam::RegHam(const std::shared_ptr<abinitio::RegSAHam> & ham,
std::tuple<CL::utility::matrix<at::Tensor>, CL::utility::matrix<at::Tensor>> (*q2x)(const std::vector<at::Tensor> &))
: abinitio::RegSAHam(*ham) {
    std::tie(xs_, JxrTs_) = q2x(qs_);
    for (size_t i = 0; i < JxrTs_.size(0); i++)
    for (size_t j = i; j < JxrTs_.size(1); j++)
    JxrTs_[i][j] = JqrT_.mm(JxrTs_[i][j]);
}
RegHam::~RegHam() {}

const CL::utility::matrix<at::Tensor> & RegHam::xs() const {return xs_;};
const CL::utility::matrix<at::Tensor> & RegHam::JxrTs() const {return JxrTs_;};





DegHam::DegHam() {}
DegHam::DegHam(const std::shared_ptr<abinitio::DegSAHam> & ham,
std::tuple<CL::utility::matrix<at::Tensor>, CL::utility::matrix<at::Tensor>> (*q2x)(const std::vector<at::Tensor> &))
: abinitio::DegSAHam(*ham) {
    std::tie(xs_, JxrTs_) = q2x(qs_);
    for (size_t i = 0; i < JxrTs_.size(0); i++)
    for (size_t j = i; j < JxrTs_.size(1); j++)
    JxrTs_[i][j] = JqrT_.mm(JxrTs_[i][j]);
}
DegHam::~DegHam() {}

const CL::utility::matrix<at::Tensor> & DegHam::xs() const {return xs_;};
const CL::utility::matrix<at::Tensor> & DegHam::JxrTs() const {return JxrTs_;};
