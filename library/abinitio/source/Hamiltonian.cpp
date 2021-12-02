#include <tchem/chemistry.hpp>

#include <abinitio/Hamiltonian.hpp>

namespace abinitio {

RegHam::RegHam() {}
RegHam::RegHam(const HamLoader & loader) : Energy(loader),
dH_(loader.dH.clone()) {
    size_t NStates = energy_.size(0);
    // convert nonadiabatic coupling to ▽H
    for (size_t i = 0    ; i < NStates; i++)
    for (size_t j = i + 1; j < NStates; j++)
    dH_[i][j] *= energy_[j] - energy_[i];
    // ▽H weight
    weight_dH_.resize(NStates);
    weight_dH_ = weight_;
    sqrtweight_dH_.resize(NStates);
    sqrtweight_dH_ = sqrtweight_;
}
RegHam::~RegHam() {}

const at::Tensor & RegHam::dH() const {return dH_;}

const double & RegHam::weight_dH(const size_t & row, const size_t & column) const {return weight_dH_[row][column];}
const double & RegHam::sqrtweight_dH(const size_t & row, const size_t & column) const {return sqrtweight_dH_[row][column];}

void RegHam::set_weight(const double & _weight) {
    Energy::set_weight(_weight);
    weight_dH_ = weight_;
    sqrtweight_dH_ = sqrtweight_;
}
void RegHam::to(const c10::DeviceType & device) {
    Energy::to(device);
    dH_.to(device);
}

// lower the energy weight for each state who has (energy - E_ref) > E_thresh
// lower the gradient weight for each gradient who has norm > dH_thresh
void RegHam::adjust_weight(const std::vector<std::pair<double, double>> & E_ref_thresh, const double & dH_thresh) {
    Energy::adjust_weight(E_ref_thresh);
    // ▽H weight
    int64_t NStates = energy_.size(0);
    for (int64_t i = 0; i < NStates; i++)
    for (int64_t j = i; j < NStates; j++) {
        double g = dH_[i][j].abs().max().item<double>();
        if (g > dH_thresh) {
            sqrtweight_dH_[i][j] = sqrtweight_ * dH_thresh / g;
            weight_dH_[i][j] = sqrtweight_dH_[i][j] * sqrtweight_dH_[i][j];
        }
    }
}





DegHam::DegHam() {}
DegHam::DegHam(const HamLoader & loader) : RegHam(loader) {
    H_ = energy_.clone();
    tchem::chem::composite_representation_(H_, dH_);
    // H weight
    size_t NStates = H_.size(0);
    weight_H_.resize(NStates);
    weight_H_ = weight_;
    sqrtweight_H_.resize(NStates);
    sqrtweight_H_ = sqrtweight_;
}
DegHam::~DegHam() {}

const at::Tensor & DegHam::H() const {return H_;};

const double & DegHam::weight_H(const size_t & row, const size_t & column) const {return weight_H_[row][column];}
const double & DegHam::sqrtweight_H(const size_t & row, const size_t & column) const {return sqrtweight_H_[row][column];}

void DegHam::set_weight(const double & _weight) {
    RegHam::set_weight(_weight);
    weight_H_ = weight_;
    sqrtweight_H_ = sqrtweight_;
}
void DegHam::to(const c10::DeviceType & device) {
    RegHam::to(device);
    H_.to(device);
}

// subtract zero point from energy and H
void DegHam::subtract_ZeroPoint(const double & zero_point) {
    RegHam::subtract_ZeroPoint(zero_point);
    H_ -= zero_point * at::eye(H_.size(0), H_.options());
}
// lower the Hamiltonian diagonal weight as energy, does not decrease off-diagonal weight
void DegHam::adjust_weight(const std::vector<std::pair<double, double>> & E_ref_thresh, const double & dH_thresh) {
    RegHam::adjust_weight(E_ref_thresh, dH_thresh);
    // H weight
    int64_t NStates = H_.size(0);
    for (int64_t i = 0; i < NStates; i++) {
        const double & ref    = E_ref_thresh[i].first ,
                     & thresh = E_ref_thresh[i].second;
        double h = H_[i][i].item<double>() - ref;
        if (h > thresh) {
            sqrtweight_H_[i][i] = sqrtweight_ * thresh / h;
            weight_H_[i][i] = sqrtweight_H_[i][i] * sqrtweight_H_[i][i];
        }
    }
}

} // namespace abinitio