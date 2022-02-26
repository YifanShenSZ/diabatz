#include "../include/data_classes.hpp"

Energy::Energy() {}
Energy::Energy(const std::shared_ptr<abinitio::SAEnergy> & ham,
std::tuple<CL::utility::matrix<at::Tensor>, CL::utility::matrix<at::Tensor>> (*q2x)(const std::vector<at::Tensor> &))
: abinitio::SAEnergy(*ham) {
    std::tie(xs_, JxqTs_) = q2x(qs_);
}
Energy::~Energy() {}

const CL::utility::matrix<at::Tensor> & Energy::xs() const {return xs_;};
const CL::utility::matrix<at::Tensor> & Energy::JxqTs() const {return JxqTs_;};

void Energy::scale_features(const CL::utility::matrix<at::Tensor> & avg, const CL::utility::matrix<at::Tensor> & std) {
    xs_ -= avg;
    xs_ /= std;
    JxqTs_ /= std;
}





RegHam::RegHam() {}
RegHam::RegHam(const std::shared_ptr<abinitio::RegSAHam> & ham,
std::tuple<CL::utility::matrix<at::Tensor>, CL::utility::matrix<at::Tensor>> (*q2x)(const std::vector<at::Tensor> &))
: abinitio::RegSAHam(*ham) {
    std::tie(xs_, JxqTs_) = q2x(qs_);
}
RegHam::~RegHam() {}

const CL::utility::matrix<at::Tensor> & RegHam::xs() const {return xs_;};
const CL::utility::matrix<at::Tensor> & RegHam::JxqTs() const {return JxqTs_;};

void RegHam::scale_features(const CL::utility::matrix<at::Tensor> & avg, const CL::utility::matrix<at::Tensor> & std) {
    xs_ -= avg;
    xs_ /= std;
    JxqTs_ /= std;
}





DegHam::DegHam() {}
DegHam::DegHam(const std::shared_ptr<abinitio::DegSAHam> & ham,
std::tuple<CL::utility::matrix<at::Tensor>, CL::utility::matrix<at::Tensor>> (*q2x)(const std::vector<at::Tensor> &))
: abinitio::DegSAHam(*ham) {
    std::tie(xs_, JxqTs_) = q2x(qs_);
}
DegHam::~DegHam() {}

const CL::utility::matrix<at::Tensor> & DegHam::xs() const {return xs_;};
const CL::utility::matrix<at::Tensor> & DegHam::JxqTs() const {return JxqTs_;};

void DegHam::scale_features(const CL::utility::matrix<at::Tensor> & avg, const CL::utility::matrix<at::Tensor> & std) {
    xs_ -= avg;
    xs_ /= std;
    JxqTs_ /= std;
}
