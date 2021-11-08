#ifndef obnet_symat_hpp
#define obnet_symat_hpp

#include <torch/torch.h>

#include <CppLibrary/utility.hpp>

namespace obnet {

// The module to hold the neural network for a symmetric matrix,
// which is actually a collection of its scalar elements
// For unknown reason (maybe hook registration) you have to call
// torch::nn::Module methods directly on symat.elements rather than symat,
// e.g. symat.elements.to(torch::kFloat64) gives what you want
// but symat.to(torch::kFloat64) does nothing
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
        symat(const std::shared_ptr<symat> & source);
        symat(const std::string & symat_in);
        ~symat();

        const int64_t & NStates() const;
        const CL::utility::matrix<size_t> & irreds() const;

        CL::utility::matrix<std::vector<at::Tensor>> parameters();

        void copy_(const std::shared_ptr<symat> & source);

        void freeze(const size_t & NLayers = -1);

        at::Tensor forward(const CL::utility::matrix<at::Tensor> & xs);
        at::Tensor operator()(const CL::utility::matrix<at::Tensor> & xs);

        // output hidden layer values before activation to `os`
        void diagnostic(const CL::utility::matrix<at::Tensor> & xs, std::ostream & os);
};

} // namespace obnet

#endif