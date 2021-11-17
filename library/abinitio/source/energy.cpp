#include <abinitio/energy.hpp>

namespace abinitio {

Energy::Energy() {}
Energy::Energy(const EnergyLoader & loader) : Geometry(loader),
energy_(loader.energy.clone()) {
    size_t NStates = energy_.size(0);
    // energy weight
    weight_E_.resize(NStates);
    std::fill(weight_E_.begin(), weight_E_.end(), 1.0);
    sqrtweight_E_.resize(NStates);
    std::fill(sqrtweight_E_.begin(), sqrtweight_E_.end(), 1.0);
}
Energy::~Energy() {}

const at::Tensor & Energy::energy() const {return energy_;}

size_t Energy::NStates() const {return energy_.size(0);}
const double & Energy::weight_E(const size_t & index) const {return weight_E_[index];}
const double & Energy::sqrtweight_E(const size_t & index) const {return sqrtweight_E_[index];}

void Energy::to(const c10::DeviceType & device) {
    Geometry::to(device);
    energy_.to(device);
}

// subtract zero point from energy
void Energy::subtract_ZeroPoint(const double & zero_point) {
    energy_ -= zero_point;
}
// lower the energy weight for each state who has (energy - E_ref) > E_thresh
void Energy::adjust_weight(const std::vector<std::pair<double, double>> & E_ref_thresh) {
    int64_t NStates = energy_.size(0);
    // energy weight
    if (E_ref_thresh.size() < NStates) throw std::invalid_argument(
    "abinitio::Energy::adjust_weight: each state must have a reference and a threshold");
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