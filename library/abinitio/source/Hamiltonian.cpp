#include <tchem/chemistry.hpp>

#include <abinitio/Hamiltonian.hpp>

namespace abinitio {

RegHam::RegHam() {}
RegHam::RegHam(const HamLoader & loader) : Geometry(loader.geom),
energy_(loader.energy.clone()), dH_(loader.dH.clone()) {
    for (size_t i = 0    ; i < dH_.size(0); i++)
    for (size_t j = i + 1; j < dH_.size(1); j++)
    dH_[i][j] *= energy_[j] - energy_[i];

    size_t NStates = energy_.size(0);
    weight_E_.resize(NStates);
    std::fill(weight_E_.begin(), weight_E_.end(), 1.0);
    sqrtweight_E_.resize(NStates);
    std::fill(sqrtweight_E_.begin(), sqrtweight_E_.end(), 1.0);
    weight_dH_.resize(NStates);
    weight_dH_ = 1.0;
    sqrtweight_dH_.resize(NStates);
    sqrtweight_dH_ = 1.0;
}
RegHam::~RegHam() {}

const at::Tensor & RegHam::energy() const {return energy_;}
const at::Tensor & RegHam::dH() const {return dH_;}

size_t RegHam::NStates() const {return energy_.size(0);}
const double & RegHam::weight_E(const size_t & index) const {return weight_E_[index];}
const double & RegHam::sqrtweight_E(const size_t & index) const {return sqrtweight_E_[index];}
const double & RegHam::weight_dH(const size_t & row, const size_t & column) const {return weight_dH_[row][column];}
const double & RegHam::sqrtweight_dH(const size_t & row, const size_t & column) const {return sqrtweight_dH_[row][column];}

void RegHam::to(const c10::DeviceType & device) {
    Geometry::to(device);
    energy_.to(device);
    dH_    .to(device);
}

// Subtract zero point from energy
void RegHam::subtract_ZeroPoint(const double & zero_point) {
    energy_ -= zero_point;
}
// Lower the weight if energy > E_thresh or ||dH|| > dH_thresh
void RegHam::adjust_weight(const double & E_thresh, const double & dH_thresh) {
    int64_t NStates = energy_.size(0);
    for (int64_t i = 0; i < NStates; i++) {
        double e = energy_[i].item<double>();
        if (e > E_thresh) {
            sqrtweight_E_[i] = E_thresh / e;
            weight_E_[i] = sqrtweight_E_[i] * sqrtweight_E_[i];
        }
    }
    for (int64_t i = 0; i < NStates; i++)
    for (int64_t j = i; j < NStates; j++) {
        double g = dH_[i][j].norm().item<double>();
        if (g > dH_thresh) {
            sqrtweight_dH_[i][j] = dH_thresh / g;
            weight_dH_[i][j] = sqrtweight_dH_[i][j] * sqrtweight_dH_[i][j];
        }
    }
}





DegHam::DegHam() {}
DegHam::DegHam(const HamLoader & loader) : RegHam(loader) {
    H_ = energy_.clone();
    tchem::chem::composite_representation_(H_, dH_);

    size_t NStates = H_.size(0);
    weight_H_.resize(NStates);
    weight_H_ = 1.0;
    sqrtweight_H_.resize(NStates);
    sqrtweight_H_ = 1.0;
}
DegHam::~DegHam() {}

const at::Tensor & DegHam::H() const {return H_;};

const double & DegHam::weight_H(const size_t & row, const size_t & column) const {return weight_H_[row][column];}
const double & DegHam::sqrtweight_H(const size_t & row, const size_t & column) const {return sqrtweight_H_[row][column];}

void DegHam::to(const c10::DeviceType & device) {
    RegHam::to(device);
    H_.to(device);
}

// Subtract zero point from energy and H
void DegHam::subtract_ZeroPoint(const double & zero_point) {
    RegHam::subtract_ZeroPoint(zero_point);
    H_ -= zero_point * at::eye(H_.size(0), H_.options());
}
// Lower the weight if H > H_thresh or ||dH|| > dH_thresh
void DegHam::adjust_weight(const double & H_thresh, const double & dH_thresh) {
    RegHam::adjust_weight(H_thresh, dH_thresh);
    int64_t NStates = energy_.size(0);
    for (int64_t i = 0; i < NStates; i++)
    for (int64_t j = i; j < NStates; j++) {
        double h = H_[i][j].item<double>();
        if (h > H_thresh) {
            sqrtweight_H_[i][j] = H_thresh / h;
            weight_H_[i][j] = sqrtweight_H_[i][j] * sqrtweight_H_[i][j];
        }
    }
}

} // namespace abinitio