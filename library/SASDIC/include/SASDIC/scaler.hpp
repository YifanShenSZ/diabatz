#ifndef SASDIC_scaler_hpp
#define SASDIC_scaler_hpp

#include <torch/torch.h>

namespace SASDIC {

// scale a dimensionless internal coordinate
class Scaler {
    private:
        size_t self_;
        std::string scaling_function_;
        std::vector<double> parameters_;
    public:
        Scaler();
        Scaler(const size_t & _self, const std::string & _scaling_function, const std::vector<double> & _parameters);
        // construct from an input line of "self    scaling_function    parameters"
        Scaler(const std::string & line);
        ~Scaler();

        const size_t & self() const;

        // given dimensionless internal coordinate,
        // return scaled dimensionless internal coordinate
        at::Tensor operator()(const at::Tensor & DIC) const;
};

} // namespace SASDIC

#endif