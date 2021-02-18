#ifndef abinitio_SAHamiltonian_hpp
#define abinitio_SAHamiltonian_hpp

#include <CppLibrary/utility.hpp>

#include <abinitio/SAloader.hpp>
#include <abinitio/SAgeometry.hpp>

namespace abinitio {

// Store regular Hamiltonian and gradient in adiabatic representation
class RegSAHam : public SAGeometry {
    protected:
        double weight_ = 1.0;
        // By regular Hamiltonian I mean energy
        at::Tensor energy_;
        // nonzero segment of ▽H elements
        CL::utility::matrix<at::Tensor> dH_;

        // point group irreducible of each matrix element
        CL::utility::matrix<size_t> irreds_;

        // Transform a Cartesian coordinate ▽H to `dH_`,
        // by finding the nonzero segment determine `irreds_`
        void construct_dH_(const at::Tensor & cartdH);
    public:
        RegSAHam();
        // See the base class constructor for details of `cart2int`
        RegSAHam(const SAHamLoader & loader,
                 std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> (*cart2int)(const at::Tensor &));
        ~RegSAHam();

        double weight() const;
        at::Tensor energy() const;
        CL::utility::matrix<size_t> irreds() const;
        CL::utility::matrix<at::Tensor> dH() const;

        void to(const c10::DeviceType & device);

        // Subtract zero point from energy
        void subtract_ZeroPoint(const double & zero_point);
        // Lower the weight if energy[0] > thresh
        void adjust_weight(const double & thresh);
};

// Store degenerate Hamiltonian and gradient in composite representation
class DegSAHam : public RegSAHam {
    protected:
        at::Tensor H_;
    public:
        DegSAHam();
        // See the base class constructor for details of `cart2int`
        DegSAHam(const SAHamLoader & loader,
                 std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> (*cart2int)(const at::Tensor &));
        ~DegSAHam();

        at::Tensor H() const;

        void to(const c10::DeviceType & device);

        // Subtract zero point from energy and H
        void subtract_ZeroPoint(const double & zero_point);
};

} // namespace abinitio

#endif