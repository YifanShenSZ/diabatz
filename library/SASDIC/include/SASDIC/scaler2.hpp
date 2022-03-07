#ifndef SASDIC_scaler2_hpp
#define SASDIC_scaler2_hpp

#include <torch/torch.h>

namespace SASDIC {

// scale a dimensionless internal coordinate with 2 scalers
class Scaler2 {
    private:
        size_t self_, other1_, other2_;
        std::string type_;
        std::vector<double> parameters_;
    public:
        Scaler2();
        Scaler2(const size_t & _self, const size_t & _other1, const size_t & _other2, const std::string & _type, const std::vector<double> & _parameters);
        // construct from an input line of "self    other1    other2    type    parameter(s)"
        Scaler2(const std::string & line);
        ~Scaler2();

        const size_t & self() const;
        const size_t & other1() const;
        const size_t & other2() const;

        // given dimensionless internal coordinates,
        // return scaled dimensionless internal coordinate
        at::Tensor operator()(const at::Tensor & DICs) const;
};

} // namespace SASDIC

#endif