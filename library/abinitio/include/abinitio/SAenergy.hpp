#ifndef abinitio_SAEnergy_hpp
#define abinitio_SAEnergy_hpp

#include <abinitio/SAloader.hpp>
#include <abinitio/SAgeometry.hpp>

namespace abinitio {

class SAEnergy : public SAGeometry {
    protected:
        at::Tensor energy_;

        // energy weight, default = 1
        std::vector<double> weight_E_, sqrtweight_E_;
    public:
        SAEnergy();
        SAEnergy(const SAEnergy & source);
        // See the base class constructor for details of `cart2CNPI`
        SAEnergy(const SAEnergyLoader & loader,
                 std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> (*cart2CNPI)(const at::Tensor &));
        ~SAEnergy();

        const at::Tensor & energy() const;

        size_t NStates() const;
        const double & weight_E(const size_t & state) const;
        const double & sqrtweight_E(const size_t & state) const;

        void set_weight(const double & _weight);
        void to(const c10::DeviceType & device);

        // subtract zero point from energy
        void subtract_ZeroPoint(const double & zero_point);
        // lower the energy weight for each state who has (energy - E_ref) > E_thresh
        void adjust_weight(const std::vector<std::pair<double, double>> & E_ref_thresh);
};

} // namespace abinitio

#endif