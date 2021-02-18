#ifndef Hd_hpp
#define Hd_hpp

#include <CppLibrary/utility.hpp>

#include <tchem/SApolynomial.hpp>

#include <obnet/symat.hpp>

// To generate input layers for Hd network
class InputGenerator {
    private:
        CL::utility::matrix<tchem::polynomial::SAPSet> polynomials_;
    public:
        InputGenerator();
        InputGenerator(const size_t & NStates, const std::vector<std::string> & sapoly_files);
        ~InputGenerator();

        CL::utility::matrix<tchem::polynomial::SAPSet> polynomials() const;

        CL::utility::matrix<at::Tensor> operator()(const std::vector<at::Tensor> & qs) const;
};

#endif