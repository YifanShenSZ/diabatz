#include "../include/data_classes.hpp"

Energy::Energy() {}
Energy::Energy(const std::shared_ptr<abinitio::SAEnergy> & ham,
std::tuple<CL::utility::matrix<at::Tensor>, CL::utility::matrix<at::Tensor>> (*q2x)(const std::vector<at::Tensor> &),
const std::shared_ptr<Hd::kernel> & pretrained_Hdkernel)
: abinitio::SAEnergy(*ham) {
    std::tie(xs_, JxrTs_) = q2x(qs_);
    for (size_t i = 0; i < JxrTs_.size(0); i++)
    for (size_t j = i; j < JxrTs_.size(1); j++)
    JxrTs_[i][j] = JqrT_.mm(JxrTs_[i][j]);
    pretrained_Hd_ = (*pretrained_Hdkernel)(qs_);
}
Energy::~Energy() {}

const CL::utility::matrix<at::Tensor> & Energy::xs() const {return xs_;};
const CL::utility::matrix<at::Tensor> & Energy::JxrTs() const {return JxrTs_;};
const at::Tensor & Energy::pretrained_Hd() const {return pretrained_Hd_;}

void Energy::scale_features(const CL::utility::matrix<at::Tensor> & avg, const CL::utility::matrix<at::Tensor> & std) {
    xs_ -= avg;
    xs_ /= std;
    JxrTs_ /= std;
}





RegHam::RegHam() {}
RegHam::RegHam(const std::shared_ptr<abinitio::RegSAHam> & ham,
std::tuple<CL::utility::matrix<at::Tensor>, CL::utility::matrix<at::Tensor>> (*q2x)(const std::vector<at::Tensor> &),
const std::shared_ptr<Hd::kernel> & pretrained_Hdkernel)
: abinitio::RegSAHam(*ham) {
    std::tie(xs_, JxrTs_) = q2x(qs_);
    for (size_t i = 0; i < JxrTs_.size(0); i++)
    for (size_t j = i; j < JxrTs_.size(1); j++)
    JxrTs_[i][j] = JqrT_.mm(JxrTs_[i][j]);
    std::tie(pretrained_Hd_, pretrained_DrHd_) = pretrained_Hdkernel->compute_Hd_dHd(geom_);
}
RegHam::~RegHam() {}

const CL::utility::matrix<at::Tensor> & RegHam::xs() const {return xs_;};
const CL::utility::matrix<at::Tensor> & RegHam::JxrTs() const {return JxrTs_;};
const at::Tensor & RegHam::pretrained_Hd  () const {return pretrained_Hd_  ;}
const at::Tensor & RegHam::pretrained_DrHd() const {return pretrained_DrHd_;}

void RegHam::scale_features(const CL::utility::matrix<at::Tensor> & avg, const CL::utility::matrix<at::Tensor> & std) {
    xs_ -= avg;
    xs_ /= std;
    JxrTs_ /= std;
}





DegHam::DegHam() {}
DegHam::DegHam(const std::shared_ptr<abinitio::DegSAHam> & ham,
std::tuple<CL::utility::matrix<at::Tensor>, CL::utility::matrix<at::Tensor>> (*q2x)(const std::vector<at::Tensor> &),
const std::shared_ptr<Hd::kernel> & pretrained_Hdkernel)
: abinitio::DegSAHam(*ham) {
    std::tie(xs_, JxrTs_) = q2x(qs_);
    for (size_t i = 0; i < JxrTs_.size(0); i++)
    for (size_t j = i; j < JxrTs_.size(1); j++)
    JxrTs_[i][j] = JqrT_.mm(JxrTs_[i][j]);
    std::tie(pretrained_Hd_, pretrained_DrHd_) = pretrained_Hdkernel->compute_Hd_dHd(geom_);
}
DegHam::~DegHam() {}

const CL::utility::matrix<at::Tensor> & DegHam::xs() const {return xs_;};
const CL::utility::matrix<at::Tensor> & DegHam::JxrTs() const {return JxrTs_;};
const at::Tensor & DegHam::pretrained_Hd  () const {return pretrained_Hd_  ;}
const at::Tensor & DegHam::pretrained_DrHd() const {return pretrained_DrHd_;}

void DegHam::scale_features(const CL::utility::matrix<at::Tensor> & avg, const CL::utility::matrix<at::Tensor> & std) {
    xs_ -= avg;
    xs_ /= std;
    JxrTs_ /= std;
}
