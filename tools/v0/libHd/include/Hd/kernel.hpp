#ifndef Hd_kernel_hpp
#define Hd_kernel_hpp

#include <SASIC/SASICSet.hpp>

#include <obnet/symat.hpp>

#include <Hd/InputGenerator.hpp>

namespace Hd {

class kernel {
    private:
        // Generate CNPI group symmetry adapted and scaled internal coordinate
        // from Cartesian coordinate
        std::shared_ptr<SASIC::SASICSet> sasicset_;
        // The neural network for Hd
        std::shared_ptr<obnet::symat> Hdnet_;
        // Generate Hd network input layer from SASIC
        std::shared_ptr<InputGenerator> input_generator_;
    public:
        kernel();
        kernel(const std::string & format, const std::string & IC, const std::string & SAS,
               const std::string & net, const std::string & checkpoint,
               const std::vector<std::string> & input_layers);
        kernel(const std::vector<std::string> & args);
        ~kernel();

        const std::shared_ptr<obnet::symat> & Hdnet() const;
        const std::shared_ptr<InputGenerator> & input_generator() const;

        size_t NStates() const;

        // Given Cartesian coordinate r, return Hd
        at::Tensor operator()(const at::Tensor & r) const;
        // Given CNPI group symmetry adapted and scaled internal coordinate, return Hd
        at::Tensor operator()(const std::vector<at::Tensor> & qs) const;

        // Given Cartesian coordinate r, return Hd and ▽Hd
        std::tuple<at::Tensor, at::Tensor> compute_Hd_dHd(const at::Tensor & r) const;
        // Given CNPI group symmetry adapted and scaled internal coordinate, return Hd and ▽Hd
        std::tuple<at::Tensor, at::Tensor> compute_Hd_dHd(const std::vector<at::Tensor> & qs) const;

        // output hidden layer values before activation to `os`
        void diagnostic(const at::Tensor & r, std::ostream & os);
};

} // namespace Hd

#endif