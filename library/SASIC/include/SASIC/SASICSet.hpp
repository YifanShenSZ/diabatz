#ifndef SASIC_SASICSet_hpp
#define SASIC_SASICSet_hpp

#include <tchem/intcoord.hpp>

#include <SASIC/SASIC.hpp>
#include <SASIC/OthScalRul.hpp>

namespace SASIC {

// A set of symmetry adapted and scaled internal coordinates
class SASICSet : public tchem::IC::IntCoordSet {
    private:
        // internal coordinate origin
        at::Tensor origin_;
        // internal coordinates who are scaled by others
        std::vector<OthScalRul> other_scaling_;
        // Internal coordinates who are scaled by themselves are picked out by self_scaling_ matrix
        // The self scaled internal coordinate vector is
        //     q = 1 - exp(-self_alpha_ * self_scaling_.mv(q))
        //       + self_complete_.mv(q)
        // This implementation trick only works for f(0) = 0, where f(x) is defined in README.md
        at::Tensor self_alpha_, self_scaling_, self_complete_;
        // sasicss_[i][j] contains the definition of
        // j-th symmetry adapted internal coordinate in i-th irreducible
        std::vector<std::vector<SASIC>> sasicss_;
    public:
        SASICSet();
        // internal coordinate definition format (Columbus7, default)
        // internal coordinate definition file
        // symmetry adaptation and scale definition file
        SASICSet(const std::string & format, const std::string & IC_file, const std::string & SAS_file);
        ~SASICSet();

        const at::Tensor & origin() const;

        // Return number of irreducible representations
        size_t NIrreds() const;
        // Return number of symmetry adapted and scaled internal coordinates per irreducible
        std::vector<size_t> NSASICs() const;
        // Return number of internal coordinates
        size_t intdim() const;

        // Return SASIC given internal coordinate q
        std::vector<at::Tensor> operator()(const at::Tensor & q);
};

} // namespace SASIC

#endif