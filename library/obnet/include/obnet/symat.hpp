#ifndef obnet_symat_hpp
#define obnet_symat_hpp

#include <torch/torch.h>

#include <CppLibrary/utility.hpp>

namespace obnet {

struct symat : torch::nn::Module {
    private:
        // number of electronc states
        int64_t NStates_;
        // CNPI group irreducible of each matrix element
        CL::utility::matrix<size_t> irreds_;
    public:
        // the upper triangle elements are stored line by line:
        // O_00, O_01, O_02, ..., O_0N, O11, O12, ..., O1N, O22, ...
        torch::nn::ModuleList elements;

        symat();
        // This copy constructor performs a somewhat deepcopy,
        // where new modules are generated and have same values as `source`
        // We do not use const reference because
        // torch::nn::ModuleList::operator[] does not support `const`,
        // although this constructor would not change `source` of course
        symat(const std::shared_ptr<symat> & source);
        symat(const std::string & symat_in);
        ~symat();

        int64_t NStates() const;
        CL::utility::matrix<size_t> irreds() const;

        void to(const torch::Dtype & dtype);
        CL::utility::matrix<std::vector<at::Tensor>> parameters();

        std::shared_ptr<symat> clone();

        at::Tensor forward(const CL::utility::matrix<at::Tensor> & xs);
        at::Tensor operator()(const CL::utility::matrix<at::Tensor> & xs);
};

} // namespace obnet

#endif