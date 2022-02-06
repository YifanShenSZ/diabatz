#include <CppLibrary/utility.hpp>

#include <SASDIC/scaler.hpp>

namespace SASDIC {

Scaler::Scaler() {}
Scaler::Scaler(const size_t & _self, const size_t & _other, const std::string & _type, const std::vector<double> & _parameters)
: self_(_self), other_(_other), type_(_type), parameters_(_parameters) {}
// construct from an input line of "self    other    type    parameter(s)"
Scaler::Scaler(const std::string & line) {
    auto strs = CL::utility::split(line);
    if (strs.size() < 3) throw(
    "SASDIC::Scaler::Scaler: wrong input line");
    self_  = std::stoul(strs[0]) - 1;
    other_ = std::stoul(strs[1]) - 1;
    type_  = strs[2];
    parameters_.resize(strs.size() - 3);
    for (size_t i = 0; i < parameters_.size(); i++) parameters_[i] = std::stod(strs[3 + i]);
}
Scaler::~Scaler() {}

const size_t & Scaler::self () const {return self_ ;}
const size_t & Scaler::other() const {return other_;}

// given dimensionless internal coordinates,
// return scaled dimensionless internal coordinate
at::Tensor Scaler::operator()(const at::Tensor & DICs) const {
    at::Tensor scaling;
    at::Tensor x = DICs[other_];
    if (type_ == "1-exp(-a*x)") {
        scaling = 1.0 - at::exp(-parameters_[0] * x);
    }
    else if (type_ == "tanh((x-a)/b)") {
        const double & a = parameters_[0],
                     & b = parameters_[1];
        scaling = at::tanh((x - a) / b);
    }
    else if (type_ == "exp(-a*x)*(1+x)^b") {
        const double & a = parameters_[0],
                     & b = parameters_[1];
        double maximum = exp(a - b) * pow(b / a, b);
        scaling = at::exp(-a * x) * (1.0 + x).pow(b) / maximum;
    }
    else throw std::invalid_argument("Unimplemented scaling function " + type_);
    if (self_ == other_) return scaling;
    else                 return scaling * DICs[self_];
}

} // namespace SASDIC