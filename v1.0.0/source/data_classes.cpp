#include "../include/data_classes.hpp"

Energy::Energy() {}
Energy::Energy(const std::shared_ptr<abinitio::SAEnergy> & ham,
std::tuple<CL::utility::matrix<at::Tensor>, CL::utility::matrix<at::Tensor>> (*q2x)(const std::vector<at::Tensor> &),
const std::shared_ptr<Hd::kernel> & pretrained_Hdkernel)
: abinitio::SAEnergy(*ham) {
    std::tie(xs_, JxqTs_) = q2x(qs_);
    pretrained_Hd_ = (*pretrained_Hdkernel)(qs_);
}
Energy::~Energy() {}

const CL::utility::matrix<at::Tensor> & Energy::xs() const {return xs_;};
const CL::utility::matrix<at::Tensor> & Energy::JxqTs() const {return JxqTs_;};
const at::Tensor & Energy::pretrained_Hd() const {return pretrained_Hd_;}





RegHam::RegHam() {}
RegHam::RegHam(const std::shared_ptr<abinitio::RegSAHam> & ham,
std::tuple<CL::utility::matrix<at::Tensor>, CL::utility::matrix<at::Tensor>> (*q2x)(const std::vector<at::Tensor> &),
const std::shared_ptr<Hd::kernel> & pretrained_Hdkernel)
: abinitio::RegSAHam(*ham) {
    std::tie(xs_, JxqTs_) = q2x(qs_);
    std::tie(pretrained_Hd_, pretrained_DqHd_) = pretrained_Hdkernel->compute_Hd_dHd(qs_);
}
RegHam::~RegHam() {}

const CL::utility::matrix<at::Tensor> & RegHam::xs() const {return xs_;};
const CL::utility::matrix<at::Tensor> & RegHam::JxqTs() const {return JxqTs_;};
const at::Tensor & RegHam::pretrained_Hd  () const {return pretrained_Hd_  ;}
const at::Tensor & RegHam::pretrained_DqHd() const {return pretrained_DqHd_;}





DegHam::DegHam() {}
DegHam::DegHam(const std::shared_ptr<abinitio::DegSAHam> & ham,
std::tuple<CL::utility::matrix<at::Tensor>, CL::utility::matrix<at::Tensor>> (*q2x)(const std::vector<at::Tensor> &),
const std::shared_ptr<Hd::kernel> & pretrained_Hdkernel)
: abinitio::DegSAHam(*ham) {
    std::tie(xs_, JxqTs_) = q2x(qs_);
    std::tie(pretrained_Hd_, pretrained_DqHd_) = pretrained_Hdkernel->compute_Hd_dHd(qs_);
}
DegHam::~DegHam() {}

const CL::utility::matrix<at::Tensor> & DegHam::xs() const {return xs_;};
const CL::utility::matrix<at::Tensor> & DegHam::JxqTs() const {return JxqTs_;};
const at::Tensor & DegHam::pretrained_Hd  () const {return pretrained_Hd_  ;}
const at::Tensor & DegHam::pretrained_DqHd() const {return pretrained_DqHd_;}
