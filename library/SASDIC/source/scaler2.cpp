#include <CppLibrary/utility.hpp>

#include <SASDIC/scaler2.hpp>

namespace SASDIC {

Scaler2::Scaler2() {}
Scaler2::Scaler2(const size_t & _self, const size_t & _other1, const size_t & _other2, const std::string & _type, const std::vector<double> & _parameters)
: self_(_self), other1_(_other1), other2_(_other2), type_(_type), parameters_(_parameters) {}
// construct from an input line of "self    other    type    parameter(s)"
Scaler2::Scaler2(const std::string & line) {
    auto strs = CL::utility::split(line);
    if (strs.size() < 4) throw std::invalid_argument(
    "SASDIC::Scaler2::Scaler2: wrong input line");
    self_   = std::stoul(strs[0]) - 1;
    other1_ = std::stoul(strs[1]) - 1;
    other2_ = std::stoul(strs[2]) - 1;
    type_   = strs[3];
    parameters_.resize(strs.size() - 4);
    for (size_t i = 0; i < parameters_.size(); i++) parameters_[i] = std::stod(strs[4 + i]);
}
Scaler2::~Scaler2() {}

const size_t & Scaler2::self  () const {return self_  ;}
const size_t & Scaler2::other1() const {return other1_;}
const size_t & Scaler2::other2() const {return other2_;}

// given dimensionless internal coordinates,
// return scaled dimensionless internal coordinate
at::Tensor Scaler2::operator()(const at::Tensor & DICs) const {
    at::Tensor scaling;
    at::Tensor x = DICs[other1_],
               y = DICs[other2_];
    if (type_ == "exp[-a*(x+y)]*[(1+x)*(1+y)]^b") {
        const double & a = parameters_[0],
                     & b = parameters_[1];
        double maximum = exp(a - b) * pow(b / a, b);
        maximum *= maximum;
        scaling = at::exp(-a * (x + y)) * ((1.0 + x) * (1.0 + y)).pow(b) / maximum;
    }
    else throw std::invalid_argument("Unimplemented scaling function " + type_);
    return scaling * DICs[self_];
}

} // namespace SASDIC