#include <CppLibrary/utility.hpp>

#include <SASDIC/scaler.hpp>

namespace SASDIC {

Scaler::Scaler() {}
Scaler::Scaler(const size_t & _self, const std::string & _scaling_function, const std::vector<double> & _parameters)
: self_(_self), scaling_function_(_scaling_function), parameters_(_parameters) {}
// construct from an input line of "self    scaling_function    parameters"
Scaler::Scaler(const std::string & line) {
    auto strs = CL::utility::split(line);
    self_ = std::stoul(strs[0]) - 1;
    scaling_function_ = strs[1];
    parameters_.resize(strs.size() - 2);
    for (size_t i = 0; i < parameters_.size(); i++) parameters_[i] = std::stod(strs[2 + i]);
}
Scaler::~Scaler() {}

const size_t & Scaler::self() const {return self_;}

// given dimensionless internal coordinate,
// return scaled dimensionless internal coordinate
at::Tensor Scaler::operator()(const at::Tensor & DIC) const {
    if (scaling_function_ == "1-exp(-a*x)") {
        return 1.0 - at::exp(-parameters_[0] * DIC);
    }
    else if (scaling_function_ == "(1+x)^2*exp(-a*x)") {
        return (1.0 + DIC) * (1.0 + DIC) * at::exp(-parameters_[0] * DIC);
    }
    else throw std::invalid_argument("Unimplemented scaling function " + scaling_function_);
}

} // namespace SASDIC