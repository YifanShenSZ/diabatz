#include "InputGenerator.hpp"

InputGenerator::InputGenerator() {}
InputGenerator::InputGenerator(const size_t & NStates, const std::vector<std::string> & sapoly_files) {
    assert(("Number of input files must equal to the number of upper triangle elements",
            sapoly_files.size() == (NStates + 1) * NStates / 2));
    polynomials_.resize(NStates);
    size_t count = 0;
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = i; j < NStates; j++) {
        polynomials_[i][j] = tchem::polynomial::SAPSet(sapoly_files[count]);
        count++;
    }
}
InputGenerator::~InputGenerator() {}

CL::utility::matrix<tchem::polynomial::SAPSet> InputGenerator::polynomials() const {return polynomials_;}

CL::utility::matrix<at::Tensor> InputGenerator::operator()(const std::vector<at::Tensor> & qs) const {
    size_t NStates = polynomials_.size();
    CL::utility::matrix<at::Tensor> xs(NStates);
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = i; j < NStates; j++)
    xs[i][j] = polynomials_[i][j](qs);
    return xs;
}