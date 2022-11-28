#include "../include/data_classes.hpp"

Energy::Energy() {}
Energy::Energy(const std::shared_ptr<abinitio::SAEnergy> & ham,
std::tuple<CL::utility::matrix<at::Tensor>, CL::utility::matrix<at::Tensor>> (*q2x1)(const std::vector<at::Tensor> &),
std::tuple<CL::utility::matrix<at::Tensor>, CL::utility::matrix<at::Tensor>> (*q2x2)(const std::vector<at::Tensor> &))
: abinitio::SAEnergy(*ham) {
    // input layer 1
    std::tie(x1s_, Jx1rTs_) = q2x1(qs_);
    for (size_t i = 0; i < Jx1rTs_.size(0); i++)
    for (size_t j = i; j < Jx1rTs_.size(1); j++)
    Jx1rTs_[i][j] = JqrT_.mm(Jx1rTs_[i][j]);
    // input layer 2
    std::tie(x2s_, Jx2rTs_) = q2x2(qs_);
    for (size_t i = 0; i < Jx2rTs_.size(0); i++)
    for (size_t j = i; j < Jx2rTs_.size(1); j++)
    Jx2rTs_[i][j] = JqrT_.mm(Jx2rTs_[i][j]);
}
Energy::~Energy() {}

const CL::utility::matrix<at::Tensor> & Energy::x1s() const {return x1s_;};
const CL::utility::matrix<at::Tensor> & Energy::Jx1rTs() const {return Jx1rTs_;};
const CL::utility::matrix<at::Tensor> & Energy::x2s() const {return x2s_;};
const CL::utility::matrix<at::Tensor> & Energy::Jx2rTs() const {return Jx2rTs_;};

void Energy::scale_features(
const CL::utility::matrix<at::Tensor> & shift1, const CL::utility::matrix<at::Tensor> & width1,
const CL::utility::matrix<at::Tensor> & shift2, const CL::utility::matrix<at::Tensor> & width2) {
    // input layer 1
    for (size_t i = 0; i < x1s_.size(0); i++)
    for (size_t j = i; j < x1s_.size(1); j++) {
           x1s_[i][j] -= shift1[i][j];
           x1s_[i][j] /= width1[i][j];
        Jx1rTs_[i][j] /= width1[i][j];
    }
    // input layer 2
    for (size_t i = 0; i < x2s_.size(0); i++)
    for (size_t j = i; j < x2s_.size(1); j++) {
           x2s_[i][j] -= shift2[i][j];
           x2s_[i][j] /= width2[i][j];
        Jx2rTs_[i][j] /= width2[i][j];
    }
}





RegHam::RegHam() {}
RegHam::RegHam(const std::shared_ptr<abinitio::RegSAHam> & ham,
std::tuple<CL::utility::matrix<at::Tensor>, CL::utility::matrix<at::Tensor>> (*q2x1)(const std::vector<at::Tensor> &),
std::tuple<CL::utility::matrix<at::Tensor>, CL::utility::matrix<at::Tensor>> (*q2x2)(const std::vector<at::Tensor> &))
: abinitio::RegSAHam(*ham) {
    // input layer 1
    std::tie(x1s_, Jx1rTs_) = q2x1(qs_);
    for (size_t i = 0; i < Jx1rTs_.size(0); i++)
    for (size_t j = i; j < Jx1rTs_.size(1); j++)
    Jx1rTs_[i][j] = JqrT_.mm(Jx1rTs_[i][j]);
    // input layer 2
    std::tie(x2s_, Jx2rTs_) = q2x2(qs_);
    for (size_t i = 0; i < Jx2rTs_.size(0); i++)
    for (size_t j = i; j < Jx2rTs_.size(1); j++)
    Jx2rTs_[i][j] = JqrT_.mm(Jx2rTs_[i][j]);
}
RegHam::~RegHam() {}

const CL::utility::matrix<at::Tensor> & RegHam::x1s() const {return x1s_;};
const CL::utility::matrix<at::Tensor> & RegHam::Jx1rTs() const {return Jx1rTs_;};
const CL::utility::matrix<at::Tensor> & RegHam::x2s() const {return x2s_;};
const CL::utility::matrix<at::Tensor> & RegHam::Jx2rTs() const {return Jx2rTs_;};

void RegHam::scale_features(
const CL::utility::matrix<at::Tensor> & shift1, const CL::utility::matrix<at::Tensor> & width1,
const CL::utility::matrix<at::Tensor> & shift2, const CL::utility::matrix<at::Tensor> & width2) {
    // input layer 1
    for (size_t i = 0; i < x1s_.size(0); i++)
    for (size_t j = i; j < x1s_.size(1); j++) {
           x1s_[i][j] -= shift1[i][j];
           x1s_[i][j] /= width1[i][j];
        Jx1rTs_[i][j] /= width1[i][j];
    }
    // input layer 2
    for (size_t i = 0; i < x2s_.size(0); i++)
    for (size_t j = i; j < x2s_.size(1); j++) {
           x2s_[i][j] -= shift2[i][j];
           x2s_[i][j] /= width2[i][j];
        Jx2rTs_[i][j] /= width2[i][j];
    }
}





DegHam::DegHam() {}
DegHam::DegHam(const std::shared_ptr<abinitio::DegSAHam> & ham,
std::tuple<CL::utility::matrix<at::Tensor>, CL::utility::matrix<at::Tensor>> (*q2x1)(const std::vector<at::Tensor> &),
std::tuple<CL::utility::matrix<at::Tensor>, CL::utility::matrix<at::Tensor>> (*q2x2)(const std::vector<at::Tensor> &))
: abinitio::DegSAHam(*ham) {
    // input layer 1
    std::tie(x1s_, Jx1rTs_) = q2x1(qs_);
    for (size_t i = 0; i < Jx1rTs_.size(0); i++)
    for (size_t j = i; j < Jx1rTs_.size(1); j++)
    Jx1rTs_[i][j] = JqrT_.mm(Jx1rTs_[i][j]);
    // input layer 2
    std::tie(x2s_, Jx2rTs_) = q2x2(qs_);
    for (size_t i = 0; i < Jx2rTs_.size(0); i++)
    for (size_t j = i; j < Jx2rTs_.size(1); j++)
    Jx2rTs_[i][j] = JqrT_.mm(Jx2rTs_[i][j]);
}
DegHam::~DegHam() {}

const CL::utility::matrix<at::Tensor> & DegHam::x1s() const {return x1s_;};
const CL::utility::matrix<at::Tensor> & DegHam::Jx1rTs() const {return Jx1rTs_;};
const CL::utility::matrix<at::Tensor> & DegHam::x2s() const {return x2s_;};
const CL::utility::matrix<at::Tensor> & DegHam::Jx2rTs() const {return Jx2rTs_;};

void DegHam::scale_features(
const CL::utility::matrix<at::Tensor> & shift1, const CL::utility::matrix<at::Tensor> & width1,
const CL::utility::matrix<at::Tensor> & shift2, const CL::utility::matrix<at::Tensor> & width2) {
    // input layer 1
    for (size_t i = 0; i < x1s_.size(0); i++)
    for (size_t j = i; j < x1s_.size(1); j++) {
           x1s_[i][j] -= shift1[i][j];
           x1s_[i][j] /= width1[i][j];
        Jx1rTs_[i][j] /= width1[i][j];
    }
    // input layer 2
    for (size_t i = 0; i < x2s_.size(0); i++)
    for (size_t j = i; j < x2s_.size(1); j++) {
           x2s_[i][j] -= shift2[i][j];
           x2s_[i][j] /= width2[i][j];
        Jx2rTs_[i][j] /= width2[i][j];
    }
}
