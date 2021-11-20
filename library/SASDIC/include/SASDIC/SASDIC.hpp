#ifndef SASDIC_SASDIC_hpp
#define SASDIC_SASDIC_hpp

#include <torch/torch.h>

namespace SASDIC {

// a symmetry adapted and scaled internal coordinate
class SASDIC {
    private:
        // (linear combination coefficient, index of scaled internal coordinate) pairs
        std::vector<std::pair<double, size_t>> coeff_indices_;
    public:
        SASDIC();
        ~SASDIC();

        const std::vector<std::pair<double, size_t>> & coeff_indices() const;

        // append a (linear combination coefficient, index of scaled internal coordinate) pair
        void append(const std::pair<double, size_t> & coeff_index);
        void append(const double & coeff, const size_t & index);

        // normalize linear combination coefficients
        void normalize();

        // given scaled dimensionless internal coordinates,
        // return symmetry adapted and scaled dimensionless internal coordinate
        at::Tensor operator()(const at::Tensor & SIC) const;
};

} // namespace SASDIC

#endif