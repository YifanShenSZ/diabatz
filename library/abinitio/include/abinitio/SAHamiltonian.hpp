#ifndef abinitio_SAHamiltonian_hpp
#define abinitio_SAHamiltonian_hpp

#include <CppLibrary/utility.hpp>

#include <abinitio/SAloader.hpp>
#include <abinitio/SAgeometry.hpp>

namespace abinitio {

// Store regular Hamiltonian and gradient in adiabatic representation
class RegSAHam : public SAGeometry {
    protected:
        // By regular Hamiltonian I mean energy
        at::Tensor energy_, dH_;

        // energy weight, default = 1
        std::vector<double> weight_E_, sqrtweight_E_;
        // gradient weight, default = 1
        CL::utility::matrix<double> weight_dH_, sqrtweight_dH_;

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

        const at::Tensor & energy() const;
        const at::Tensor & dH() const;
        // point group irreducible of each matrix element
        const CL::utility::matrix<size_t> & irreds() const;
        // nonzero segment of ▽H elements in point group symmetry adapted internal coordinate
        const CL::utility::matrix<at::Tensor> & SAdH() const;

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
class DegSAHam : public RegSAHam {
    protected:
        at::Tensor H_;

        // Hamiltonian weight, default = 1
        CL::utility::matrix<double> weight_H_, sqrtweight_H_;
    public:
        DegSAHam();
        DegSAHam(const DegSAHam & source);
        // See the base class constructor for details of `cart2int`
        DegSAHam(const SAHamLoader & loader,
                 std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> (*cart2int)(const at::Tensor &));
        ~DegSAHam();

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