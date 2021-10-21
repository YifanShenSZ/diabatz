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

        // point group irreducible of each matrix element
        CL::utility::matrix<size_t> irreds_;
        // nonzero segment of ▽H elements in point group symmetry adapted internal coordinate
        CL::utility::matrix<at::Tensor> SAdH_;

        // energy weight, default = 1
        std::vector<double> weight_E_, sqrtweight_E_;
        // gradient weight, default = 1
        CL::utility::matrix<double> weight_dH_, sqrtweight_dH_;

        // Construct `SAdH_` based on constructed `dH_`
        // Determine `irreds_` by finding the largest segment of each `SAdH_` element
        void construct_symmetry_();
        // Reconstruct `dH_` based on constructed `SAdH_`
        // to eliminate the symmetry breaking flaw in original data
        void reconstruct_dH_();
    public:
        RegSAHam();
        RegSAHam(const RegSAHam & source);
        // See the base class constructor for details of `cart2CNPI`
        RegSAHam(const SAHamLoader & loader,
                 std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> (*cart2CNPI)(const at::Tensor &));
        ~RegSAHam();

        const at::Tensor & energy() const;
        const at::Tensor & dH() const;
        const size_t & irreds(const size_t & row, const size_t & column) const;
        const at::Tensor & SAdH(const size_t & row, const size_t & column) const;

        size_t NStates() const;
        const double & weight_E(const size_t & state) const;
        const double & sqrtweight_E(const size_t & state) const;
        const double & weight_dH(const size_t & row, const size_t & column) const;
        const double & sqrtweight_dH(const size_t & row, const size_t & column) const;

        void to(const c10::DeviceType & device);

        // subtract zero point from energy
        void subtract_ZeroPoint(const double & zero_point);
        // lower the energy weight for each state who has (energy - E_ref) > E_thresh
        // lower the gradient weight for each gradient who has norm > dH_thresh
        void adjust_weight(const std::vector<std::pair<double, double>> & E_ref_thresh, const double & dH_thresh);
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
        // See the base class constructor for details of `cart2CNPI`
        DegSAHam(const SAHamLoader & loader,
                 std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> (*cart2CNPI)(const at::Tensor &));
        ~DegSAHam();

        const at::Tensor & H() const;

        const double & weight_H(const size_t & row, const size_t & column) const;
        const double & sqrtweight_H(const size_t & row, const size_t & column) const;

        void to(const c10::DeviceType & device);

        // subtract zero point from energy and H
        void subtract_ZeroPoint(const double & zero_point);
        // lower the Hamiltonian diagonal weight as energy, does not decrease off-diagonal weight
        void adjust_weight(const std::vector<std::pair<double, double>> & E_ref_thresh, const double & dH_thresh);
};

} // namespace abinitio

#endif