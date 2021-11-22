#ifndef SASDIC_scaler_hpp
#define SASDIC_scaler_hpp

#include <torch/torch.h>

namespace SASDIC {

// scale a dimensionless internal coordinate
class Scaler {
    private:
        size_t self_, other_;
        std::string type_;
        std::vector<double> parameters_;
    public:
        Scaler();
        Scaler(const size_t & _self, const size_t & _other, const std::string & _type, const std::vector<double> & _parameters);
        // construct from an input line of "self    other    type    parameter(s)"
        Scaler(const std::string & line);
        ~Scaler();

        const size_t & self() const;
        const size_t & other() const;

        // given dimensionless internal coordinates,
        // return scaled dimensionless internal coordinate
        at::Tensor operator()(const at::Tensor & DICs) const;
};

} // namespace SASDIC

#endif