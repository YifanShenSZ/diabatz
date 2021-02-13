#ifndef abinitio_Hamiltonian_hpp
#define abinitio_Hamiltonian_hpp

#include <abinitio/loader.hpp>
#include <abinitio/geometry.hpp>

namespace abinitio {

// Store regular Hamiltonian and gradient in adiabatic representation
class RegHam : public Geometry {
    protected:
        double weight_ = 1.0;
        // By regular Hamiltonian I mean energy
        at::Tensor energy_, dH_;
    public:
        RegHam();
        RegHam(const HamLoader & loader);
        ~RegHam();

        double weight() const;
        at::Tensor energy() const;
        at::Tensor dH() const;

        void to(const c10::DeviceType & device);

        // Subtract zero point from energy
        void subtract_ZeroPoint(const double & zero_point);
        // Lower the weight if energy[0] > thresh
        void adjust_weight(const double & thresh);
};

// Store degenerate Hamiltonian and gradient in composite representation
class DegHam : public RegHam {
    protected:
        at::Tensor H_;
    public:
        DegHam();
        DegHam(const HamLoader & loader);
        ~DegHam();

        at::Tensor H() const;

        void to(const c10::DeviceType & device);

        // Subtract zero point from energy and H
        void subtract_ZeroPoint(const double & zero_point);
};

} // namespace abinitio

#endif