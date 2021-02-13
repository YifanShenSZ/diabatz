#ifndef abinitio_SAHamiltonian_hpp
#define abinitio_SAHamiltonian_hpp

#include <abinitio/SAloader.hpp>
#include <abinitio/SAgeometry.hpp>

namespace abinitio {

// Store regular Hamiltonian and gradient in adiabatic representation
class RegSAHam : public SAGeometry {
    protected:
        // the metric of point group symmetry adapted internal coordinate gradient
        // S = J . J^T
        // where J is the Jacobian of internal coordinate over Cartesian coordiate
        std::vector<at::Tensor> Ss_;

        double weight_ = 1.0;
        // By regular Hamiltonian I mean energy
        at::Tensor energy_, dH_;
    public:
        RegSAHam();
        RegSAHam(const HamLoader & loader);
        ~RegSAHam();

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