#ifndef abinitio_Hamiltonian_hpp
#define abinitio_Hamiltonian_hpp

#include <CppLibrary/utility.hpp>

#include <abinitio/loader.hpp>
#include <abinitio/geometry.hpp>

namespace abinitio {

// Store regular Hamiltonian and gradient in adiabatic representation
class RegHam : public Geometry {
    protected:
        // By regular Hamiltonian I mean energy
        at::Tensor energy_, dH_;

        // energy weight, default = 1
        std::vector<double> weight_E_, sqrtweight_E_;
        // gradient weight, default = 1
        CL::utility::matrix<double> weight_dH_, sqrtweight_dH_;
    public:
        RegHam();
        RegHam(const HamLoader & loader);
        ~RegHam();

        const at::Tensor & energy() const;
        const at::Tensor & dH() const;

        size_t NStates() const;
        const double & weight_E(const size_t & index) const;
        const double & sqrtweight_E(const size_t & index) const;
        const double & weight_dH(const size_t & row, const size_t & column) const;
        const double & sqrtweight_dH(const size_t & row, const size_t & column) const;

        void to(const c10::DeviceType & device);

        // Subtract zero point from energy
        void subtract_ZeroPoint(const double & zero_point);
        // Lower the weight if energy > E_thresh or ||dH|| > dH_thresh
        void adjust_weight(const double & E_thresh, const double & dH_thresh);
};

// Store degenerate Hamiltonian and gradient in composite representation
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

        void to(const c10::DeviceType & device);

        // Subtract zero point from energy and H
        void subtract_ZeroPoint(const double & zero_point);
        // Lower the weight if H > H_thresh or ||dH|| > dH_thresh
        void adjust_weight(const double & H_thresh, const double & dH_thresh);
};

} // namespace abinitio

#endif