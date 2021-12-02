#include <tchem/chemistry.hpp>

#include <abinitio/SAHamiltonian.hpp>

namespace abinitio {

// Construct `SAdH_` based on constructed `dH_`
// Determine `irreds_` by finding the largest segment of each `SAdH_` element
void RegSAHam::construct_symmetry_() {
    size_t NStates = dH_.size(0), NIrreds = NPointIrreds();
    SAdH_  .resize(NStates);
    irreds_.resize(NStates);
    for (size_t i = 0; i < NStates; i++) {
        // diagonals must be totally symmetric
        SAdH_  [i][i] = C2Qs_[0].mv(dH_[i][i]);
        irreds_[i][i] = 0;
        // Try out every irreducible for off-diagonals
        for (size_t j = i + 1; j < NStates; j++) {
            SAdH_[i][j] = C2Qs_[0].mv(dH_[i][j]);
            double infnorm = at::max(at::abs(SAdH_[i][j])).item<double>();
            irreds_[i][j] = 0;
            for (size_t irred = 1; irred < NIrreds; irred++) {
                at::Tensor candidate = C2Qs_[irred].mv(dH_[i][j]);
                double cannorm = at::max(at::abs(candidate)).item<double>();
                if (cannorm > infnorm) {
                    SAdH_[i][j] = candidate;
                    infnorm = cannorm;
                    irreds_[i][j] = irred;
                }
            }
        }
    }
}
// Reconstruct `dH_` based on constructed `SAdH_`
// to eliminate the symmetry breaking flaw in original data
void RegSAHam::reconstruct_dH_() {
    for (size_t i = 0; i < dH_.size(0); i++)
    for (size_t j = i; j < dH_.size(1); j++)
    dH_[i][j] = JQrTs_[irreds_[i][j]].mv(SAdH_[i][j]);
}

RegSAHam::RegSAHam() {}
RegSAHam::RegSAHam(const RegSAHam & source) : SAEnergy(source), dH_(source.dH_),
irreds_(source.irreds_), SAdH_(source.SAdH_),
weight_dH_(source.weight_dH_), sqrtweight_dH_(source.sqrtweight_dH_) {}
// See the base class constructor for details of `cart2CNPI`
RegSAHam::RegSAHam(const SAHamLoader & loader,
std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> (*cart2CNPI)(const at::Tensor &))
: SAEnergy(loader, cart2CNPI), dH_(loader.dH.clone()) {
    size_t NStates = energy_.size(0);
    // convert nonadiabatic coupling to ▽H
    for (size_t i = 0    ; i < NStates; i++)
    for (size_t j = i + 1; j < NStates; j++)
    dH_[i][j] *= energy_[j] - energy_[i];
    // adapt symmetry
    this->construct_symmetry_();
    this->reconstruct_dH_();
    // ▽H weight
    weight_dH_.resize(NStates);
    weight_dH_ = weight_;
    sqrtweight_dH_.resize(NStates);
    sqrtweight_dH_ = sqrtweight_;
}
RegSAHam::~RegSAHam() {}

const at::Tensor & RegSAHam::dH() const {return dH_;}

const size_t & RegSAHam::irreds(const size_t & row, const size_t & column) const {return irreds_[row][column];}
const at::Tensor & RegSAHam::SAdH(const size_t & row, const size_t & column) const {return SAdH_[row][column];}
const double & RegSAHam::weight_dH(const size_t & row, const size_t & column) const {return weight_dH_[row][column];}
const double & RegSAHam::sqrtweight_dH(const size_t & row, const size_t & column) const {return sqrtweight_dH_[row][column];}

void RegSAHam::set_weight(const double & _weight) {
    SAEnergy::set_weight(_weight);
    weight_dH_ = weight_;
    sqrtweight_dH_ = sqrtweight_;
}
void RegSAHam::to(const c10::DeviceType & device) {
    SAEnergy::to(device);
    dH_.to(device);
    for (auto & row : SAdH_) for (auto & col : row) col.to(device);
}

// lower the energy weight for each state who has (energy - E_ref) > E_thresh
// lower the gradient weight for each gradient who has norm > dH_thresh
void RegSAHam::adjust_weight(const std::vector<std::pair<double, double>> & E_ref_thresh, const double & dH_thresh) {
    SAEnergy::adjust_weight(E_ref_thresh);
    // ▽H weight
    int64_t NStates = energy_.size(0);
    for (int64_t i = 0; i < NStates; i++)
    for (int64_t j = i; j < NStates; j++) {
        double g = dH_[i][j].abs().max().item<double>();
        if (g > dH_thresh) {
            sqrtweight_dH_[i][j] *= dH_thresh / g;
            weight_dH_[i][j] = sqrtweight_dH_[i][j] * sqrtweight_dH_[i][j];
        }
    }
}





DegSAHam::DegSAHam() {}
DegSAHam::DegSAHam(const DegSAHam & source) : RegSAHam(source),
H_(source.H_), weight_H_(source.weight_H_), sqrtweight_H_(source.sqrtweight_H_) {}
// See the base class constructor for details of `cart2CNPI`
DegSAHam::DegSAHam(const SAHamLoader & loader,
std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> (*cart2CNPI)(const at::Tensor &))
: RegSAHam(loader, cart2CNPI) {
    size_t NStates = energy_.size(0);
    // adiabatic representation -> composite representation
    H_ = energy_.clone();
    tchem::chem::composite_representation_(H_, dH_);
    // since the state ordering may have changed during a->c transformation,
    // run `construct_symmetry_` again to update `irreds_` and `SAdH_`
    this->construct_symmetry_();
    // point group symmetry consistency check
    for (size_t i = 0    ; i < NStates; i++)
    for (size_t j = i + 1; j < NStates; j++)
    if (abs(H_[i][j].item<double>()) > 1e-12 && irreds_[i][j] != 0)
    std::cerr << "Warning: inconsistent irreducible between H and ▽H\n";
    // weights
    weight_H_.resize(NStates);
    weight_H_ = weight_;
    sqrtweight_H_.resize(NStates);
    sqrtweight_H_ = sqrtweight_;
}
DegSAHam::~DegSAHam() {}

const at::Tensor & DegSAHam::H() const {return H_;};

const double & DegSAHam::weight_H(const size_t & row, const size_t & column) const {return weight_H_[row][column];}
const double & DegSAHam::sqrtweight_H(const size_t & row, const size_t & column) const {return sqrtweight_H_[row][column];}

void DegSAHam::set_weight(const double & _weight) {
    RegSAHam::set_weight(_weight);
    weight_H_ = weight_;
    sqrtweight_H_ = sqrtweight_;
}
void DegSAHam::to(const c10::DeviceType & device) {
    RegSAHam::to(device);
    H_.to(device);
}

// subtract zero point from energy and H
void DegSAHam::subtract_ZeroPoint(const double & zero_point) {
    RegSAHam::subtract_ZeroPoint(zero_point);
    H_ -= zero_point * at::eye(H_.size(0), H_.options());
}
// lower the Hamiltonian diagonal weight as energy, does not decrease off-diagonal weight
void DegSAHam::adjust_weight(const std::vector<std::pair<double, double>> & E_ref_thresh, const double & dH_thresh) {
    RegSAHam::adjust_weight(E_ref_thresh, dH_thresh);
    // H weight
    int64_t NStates = H_.size(0);
    if (E_ref_thresh.size() < NStates) throw std::invalid_argument(
    "abinitio::DegSAHam::adjust_weight: each state must have a reference and a threshold");
    for (int64_t i = 0; i < NStates; i++) {
        const double & ref    = E_ref_thresh[i].first ,
                     & thresh = E_ref_thresh[i].second;
        double h = H_[i][i].item<double>() - ref;
        if (h > thresh) {
            sqrtweight_H_[i][i] *= thresh / h;
            weight_H_[i][i] = sqrtweight_H_[i][i] * sqrtweight_H_[i][i];
        }
    }
}

} // namespace abinitio