#include <CppLibrary/linalg.hpp>

#include <SASDIC/SASDIC.hpp>

namespace SASDIC {

SASDIC::SASDIC() {}
SASDIC::~SASDIC() {}

const std::vector<std::pair<double, size_t>> & SASDIC::coeff_indices() const {return coeff_indices_;}

// Append a linear combination coefficient - index of scaled internal coordinate pair
void SASDIC::append(const std::pair<double, size_t> & coeff_index) {
    coeff_indices_.push_back(coeff_index);
}
void SASDIC::append(const double & coeff, const size_t & index) {
    coeff_indices_.push_back(std::pair<double, size_t>(coeff, index));
}

// Normalize linear combination coefficients
void SASDIC::normalize() {
    double norm2 = 0.0;
    for (const auto & coeff_index : coeff_indices_) norm2 += coeff_index.first * coeff_index.first;
    norm2 /= sqrt(norm2);
    for (auto & coeff_index : coeff_indices_) coeff_index.first /= norm2;
}

// given scaled dimensionless internal coordinates,
// return symmetry adapted and scaled dimensionless internal coordinate
at::Tensor SASDIC::operator()(const at::Tensor & SIC) const {
    at::Tensor sasdic = coeff_indices_[0].first * SIC[coeff_indices_[0].second];
    for (size_t i = 1; i < coeff_indices_.size(); i++)
    sasdic = sasdic + coeff_indices_[i].first * SIC[coeff_indices_[i].second];
    return sasdic;
}

} // namespace SASDIC