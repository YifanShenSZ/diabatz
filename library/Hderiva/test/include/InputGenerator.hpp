#ifndef InputGenerator_hpp
#define InputGenerator_hpp

#include <CppLibrary/utility.hpp>

#include <tchem/polynomial.hpp>

#include <obnet/symat.hpp>

// To generate input layers for Hd network
class InputGenerator {
    private:
        CL::utility::matrix<tchem::polynomial::SAPSet> polynomials_;
    public:
        InputGenerator();
        InputGenerator(const size_t & NStates, const CL::utility::matrix<size_t> & irreds, const std::vector<std::string> & sapoly_files, const std::vector<size_t> & dimensions);
        ~InputGenerator();

        const CL::utility::matrix<tchem::polynomial::SAPSet> & polynomials() const;

        CL::utility::matrix<at::Tensor> operator()(const std::vector<at::Tensor> & qs) const;
        std::tuple<CL::utility::matrix<at::Tensor>, CL::utility::matrix<at::Tensor>> compute_x_JT(const std::vector<at::Tensor> & qs) const;
        CL::utility::matrix<at::Tensor> compute_K(const std::vector<at::Tensor> & qs) const;
};

#endif