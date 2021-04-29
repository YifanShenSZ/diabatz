#include "../include/InputGenerator.hpp"

InputGenerator::InputGenerator() {}
InputGenerator::InputGenerator(const size_t & NStates, const CL::utility::matrix<size_t> & irreds, const std::vector<std::string> & sapoly_files, const std::vector<size_t> & dimensions) {
    if (sapoly_files.size() != (NStates + 1) * NStates / 2) throw std::invalid_argument(
    "InputGenerator::InputGenerator: The number of input files must equal to the number of upper triangle elements");
    polynomials_.resize(NStates);
    size_t count = 0;
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = i; j < NStates; j++) {
        polynomials_[i][j] = tchem::polynomial::SAPSet(sapoly_files[count], irreds[i][j],dimensions);
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
std::tuple<CL::utility::matrix<at::Tensor>, CL::utility::matrix<at::Tensor>>
InputGenerator::compute_x_JT(const std::vector<at::Tensor> & qs) const {
    size_t NStates = polynomials_.size();
    CL::utility::matrix<at::Tensor> xs(NStates), JTs(NStates);
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = i; j < NStates; j++) {
        xs[i][j] = polynomials_[i][j](qs);
        std::vector<at::Tensor> Js = polynomials_[i][j].Jacobian(qs);
        for (at::Tensor & J : Js) J.transpose_(0, 1);
        JTs[i][j] = at::cat(Js);
    }
    return std::make_tuple(xs, JTs);
}