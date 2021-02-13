#ifndef obnet_symat_hpp
#define obnet_symat_hpp

#include <torch/torch.h>

#include <CppLibrary/utility.hpp>

namespace obnet {

struct symat : torch::nn::Module {
    private:
        // number of electronc states
        int64_t NStates_;
        // the CNPI group symmetry (irreducible) of each matrix element
        CL::utility::matrix<size_t> symmetry_;
    public:
        // the upper triangle elements are stored line by line:
        // O_00, O_01, O_02, ..., O_0N, O11, O12, ..., O1N, O22, ...
        torch::nn::ModuleList elements;

        symat();
        symat(const std::string & symat_in);
        ~symat();

        int64_t NStates() const;
        CL::utility::matrix<size_t> symmetry() const;

        void to(const torch::Dtype & dtype);
        CL::utility::matrix<std::vector<at::Tensor>> parameters();

        at::Tensor forward(const CL::utility::matrix<at::Tensor> & xs);
        at::Tensor operator()(const CL::utility::matrix<at::Tensor> & xs);
};

} // namespace obnet

#endif