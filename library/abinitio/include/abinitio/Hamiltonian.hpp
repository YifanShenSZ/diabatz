#ifndef abinitio_Hamiltonian_hpp
#define abinitio_Hamiltonian_hpp

#include <CppLibrary/utility.hpp>

#include <abinitio/loader.hpp>
#include <abinitio/energy.hpp>

namespace abinitio {

// store regular Hamiltonian and gradient in adiabatic representation
class RegHam : public Energy {
    protected:
        // regular Hamiltonian (energy) is in Energy
        at::Tensor dH_;

        // gradient weight, default = 1
        CL::utility::matrix<double> weight_dH_, sqrtweight_dH_;
    public:
        RegHam();
        RegHam(const HamLoader & loader);
        ~RegHam();

        const at::Tensor & dH() const;

        const double & weight_dH(const size_t & row, const size_t & column) const;
        const double & sqrtweight_dH(const size_t & row, const size_t & column) const;

        void set_weight(const double & _weight);
        void to(const c10::DeviceType & device);

        // lower the energy weight for each state who has (energy - E_ref) > E_thresh
        // lower the gradient weight for each gradient who has norm > dH_thresh
        void adjust_weight(const std::vector<std::pair<double, double>> & E_ref_thresh, const double & dH_thresh);
};

// store degenerate Hamiltonian and gradient in composite representation
class DegHam : public RegHam {
    protected:
        at::Tensor H_;

        // Hamiltonian weight, default = 1
        CL::utility::matrix<double> weight_H_, sqrtweight_H_;
    public:
        DegHam();
        DegHam(const HamLoader & loader);
        ~DegHam();

        const at::Tensor & H() const;

        const double & weight_H(const size_t & row, const size_t & column) const;
        const double & sqrtweight_H(const size_t & row, const size_t & column) const;

        void set_weight(const double & _weight);
        void to(const c10::DeviceType & device);

        // subtract zero point from energy and H
        void subtract_ZeroPoint(const double & zero_point);
        // lower the Hamiltonian diagonal weight as energy, does not decrease off-diagonal weight
        void adjust_weight(const std::vector<std::pair<double, double>> & E_ref_thresh, const double & dH_thresh);
};

} // namespace abinitio

#endif