#ifndef Hd_kernel_hpp
#define Hd_kernel_hpp

#include <SASDIC/SASDICSet.hpp>

#include <obnet/symat.hpp>

#include <Hd/InputGenerator.hpp>

namespace Hd {

class kernel {
    private:
        // generate CNPI group symmetry adapted and scaled internal coordinate from Cartesian coordinate
        std::shared_ptr<SASDIC::SASDICSet> sasicset_;
        // the neural network for Hd
        std::shared_ptr<obnet::symat> Hdnet_;
        // generate Hd network input layer from SASDIC
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

        // given Cartesian coordinate r, return Hd
        at::Tensor operator()(const at::Tensor & r) const;
        // given CNPI group symmetry adapted and scaled internal coordinate, return Hd
        at::Tensor operator()(const std::vector<at::Tensor> & qs) const;

        // given Cartesian coordinate r, return Hd and ▽Hd
        std::tuple<at::Tensor, at::Tensor> compute_Hd_dHd(const at::Tensor & r) const;
        // given CNPI group symmetry adapted and scaled internal coordinate, return Hd and ▽Hd
        std::tuple<at::Tensor, at::Tensor> compute_Hd_dHd(const std::vector<at::Tensor> & qs) const;

        // output hidden layer values before activation to `os`
        void diagnostic(const at::Tensor & r, std::ostream & os);
};

} // namespace Hd

#endif