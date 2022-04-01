#ifndef Hd_kernel_hpp
#define Hd_kernel_hpp

#include <SASDIC/SASDICSet.hpp>

#include <obnet/symat.hpp>

#include <Hd/InputGenerator.hpp>

namespace Hd {

class kernel {
    private:
        // generate CNPI group symmetry adapted and scaled internal coordinate from Cartesian coordinate
        std::shared_ptr<SASDIC::SASDICSet> sasicset1_, sasicset2_;
        // the neural network for Hd
        std::shared_ptr<obnet::symat> Hdnet1_, Hdnet2_;
        // generate Hd network input layer from SASDIC
        std::shared_ptr<InputGenerator> input_generator1_, input_generator2_;
    public:
        kernel();
        kernel(const std::string & format1, const std::string & IC1, const std::string & SAS1,
               const std::string & net1, const std::string & checkpoint1,
               const std::vector<std::string> & input_layers1,
               const std::string & format2, const std::string & IC2, const std::string & SAS2,
               const std::string & net2, const std::string & checkpoint2,
               const std::vector<std::string> & input_layers2);
        kernel(const std::vector<std::string> & args);
        ~kernel();

        size_t NStates() const;

        // given Cartesian coordinate r, return Hd
        at::Tensor operator()(const at::Tensor & r) const;

        // given Cartesian coordinate r, return Hd and ▽Hd
        std::tuple<at::Tensor, at::Tensor> compute_Hd_dHd(const at::Tensor & r) const;

        // output hidden layer values before activation to `os`
        void diagnostic(const at::Tensor & r, std::ostream & os);
};

} // namespace Hd

#endif