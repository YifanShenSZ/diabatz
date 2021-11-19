#include <tchem/chemistry.hpp>

#include <abinitio/SAenergy.hpp>

namespace abinitio {


SAEnergy::SAEnergy() {}

SAEnergy::SAEnergy(const SAEnergy & source) : SAGeometry(source), energy_(source.energy_),
weight_E_(source.weight_E_), sqrtweight_E_(source.sqrtweight_E_) {}

// See the base class constructor for details of `cart2CNPI`
SAEnergy::SAEnergy(const SAEnergyLoader & loader,
std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> (*cart2CNPI)(const at::Tensor &))
: SAGeometry(loader, cart2CNPI), energy_(loader.energy.clone()) {
    size_t NStates = energy_.size(0);
    weight_E_.resize(NStates);
    std::fill(weight_E_.begin(), weight_E_.end(), weight_);
    sqrtweight_E_.resize(NStates);
    std::fill(sqrtweight_E_.begin(), sqrtweight_E_.end(), sqrtweight_);
}

SAEnergy::~SAEnergy() {}

const at::Tensor & SAEnergy::energy() const {return energy_;}

size_t SAEnergy::NStates() const {return energy_.size(0);}
const double & SAEnergy::weight_E(const size_t & state) const {return weight_E_[state];}
const double & SAEnergy::sqrtweight_E(const size_t & state) const {return sqrtweight_E_[state];}

void SAEnergy::to(const c10::DeviceType & device) {
    SAGeometry::to(device);
    energy_.to(device);
}

// subtract zero point from energy
void SAEnergy::subtract_ZeroPoint(const double & zero_point) {
    energy_ -= zero_point;
}
// lower the energy weight for each state who has (energy - E_ref) > E_thresh
void SAEnergy::adjust_weight(const std::vector<std::pair<double, double>> & E_ref_thresh) {
    int64_t NStates = energy_.size(0);
    if (E_ref_thresh.size() < NStates) throw std::invalid_argument(
    "abinitio::SAEnergy::adjust_weight: each state must have a reference and a threshold");
    for (int64_t i = 0; i < NStates; i++) {
        const double & ref    = E_ref_thresh[i].first ,
                     & thresh = E_ref_thresh[i].second;
        double e = energy_[i].item<double>() - ref;
        if (e > thresh) sqrtweight_E_[i] = sqrtweight_ * thresh / e;
        else            sqrtweight_E_[i] = sqrtweight_;
        weight_E_[i] = sqrtweight_E_[i] * sqrtweight_E_[i];
    }
}

} // namespace abinitio