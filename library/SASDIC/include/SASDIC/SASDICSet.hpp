#ifndef SASDIC_SASDICSet_hpp
#define SASDIC_SASDICSet_hpp

#include <tchem/intcoord.hpp>

#include <SASDIC/SASDIC.hpp>
#include <SASDIC/scaler.hpp>

namespace SASDIC {

// a set of symmetry adapted and scaled dimensionless internal coordinates
class SASDICSet : public tchem::IC::IntCoordSet {
    private:
        // internal coordinate origin
        at::Tensor origin_;
        // Internal coordinates with no scaling are picked out by scaling_complete_ matrix
        // The scaled dimensionless internal coordinates vector is
        //     SDICs = scaling_complete_.mv(DICs)
        //           + pass DICs to scalers
        std::vector<Scaler> scalers_;
        at::Tensor scaling_complete_;
        // sasdicss_[i][j] contains the definition of
        // j-th symmetry adapted internal coordinate in i-th irreducible
        std::vector<std::vector<SASDIC>> sasdicss_;
    public:
        SASDICSet();
        // internal coordinate definition format (Columbus7, default)
        // internal coordinate definition file
        // symmetry adaptation and scale definition file
        SASDICSet(const std::string & format, const std::string & IC_file, const std::string & SAS_file);
        ~SASDICSet();

        const at::Tensor & origin() const;

        // number of irreducible representations
        size_t NIrreds() const;
        // number of symmetry adapted and scaled dimensionless internal coordinates per irreducible
        std::vector<size_t> NSASDICs() const;
        // number of internal coordinates
        size_t intdim() const;

        // given internal coordinates q, return SASDIC
        std::vector<at::Tensor> operator()(const at::Tensor & q) const;
};

} // namespace SASDIC

#endif