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
        at::Tensor energy_, dH_;

        // point group irreducible of each matrix element
        CL::utility::matrix<size_t> irreds_;
        // nonzero segment of ▽H elements in point group symmetry adapted internal coordinate
        CL::utility::matrix<at::Tensor> SAdH_;

        // Construct `SAdH_` based on constructed `dH_`
        // Determine `irreds_` by finding the largest segment of each `SAdH_` element
        void construct_symmetry_();
        // Reconstruct `dH_` based on constructed `SAdH_`
        // to eliminate the symmetry breaking flaw in original data
        void reconstruct_dH_();
    public:
        RegSAHam();
        RegSAHam(const RegSAHam & source);
        // See the base class constructor for details of `cart2int`
        RegSAHam(const SAHamLoader & loader,
                 std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> (*cart2int)(const at::Tensor &));
        ~RegSAHam();

        const double & weight() const;
        const at::Tensor & energy() const;
        const at::Tensor & dH() const;
        const CL::utility::matrix<size_t> & irreds() const;
        const CL::utility::matrix<at::Tensor> & SAdH() const;

        size_t NStates() const;

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
        DegSAHam(const DegSAHam & source);
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