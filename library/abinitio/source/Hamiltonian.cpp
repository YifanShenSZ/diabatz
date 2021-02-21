#include <tchem/chemistry.hpp>

#include <abinitio/Hamiltonian.hpp>

namespace abinitio {

RegHam::RegHam() {}
RegHam::RegHam(const HamLoader & loader) {
    geom_   = loader.geom  .clone();
    energy_ = loader.energy.clone();
    dH_     = loader.dH    .clone();
    for (size_t i = 0    ; i < dH_.size(0); i++)
    for (size_t j = i + 1; j < dH_.size(1); j++)
    dH_[i][j] *= energy_[j] - energy_[i];
}
RegHam::~RegHam() {}

double RegHam::weight() const {return weight_;}
at::Tensor RegHam::energy() const {return energy_;}
at::Tensor RegHam::dH() const {return dH_;}

size_t RegHam::NStates() const {return energy_.size(0);}

void RegHam::to(const c10::DeviceType & device) {
    Geometry::to(device);
    energy_.to(device);
    dH_    .to(device);
}

// Subtract zero point from energy
void RegHam::subtract_ZeroPoint(const double & zero_point) {
    energy_ -= zero_point;
}
// Lower the weight if energy[0] > thresh
void RegHam::adjust_weight(const double & thresh) {
    double temp = energy_[0].item<double>();
    if (temp > thresh) {
        temp = thresh / temp;
        weight_ = temp * temp;
    }
}





DegHam::DegHam() {}
DegHam::DegHam(const HamLoader & loader) : RegHam(loader) {
    H_ = energy_.clone();
    tchem::chem::composite_representation_(H_, dH_);
}
DegHam::~DegHam() {}

at::Tensor DegHam::H() const {return H_;};

void DegHam::to(const c10::DeviceType & device) {
    RegHam::to(device);
    H_.to(device);
}

// Subtract zero point from energy and H
void DegHam::subtract_ZeroPoint(const double & zero_point) {
    RegHam::subtract_ZeroPoint(zero_point);
    H_ -= zero_point * at::eye(H_.size(0), H_.options());
}

} // namespace abinitio