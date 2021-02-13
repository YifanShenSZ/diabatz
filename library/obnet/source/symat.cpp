#include <CppLibrary/utility.hpp>

#include <obnet/scalar.hpp>

#include <obnet/symat.hpp>

namespace obnet {

symat::symat() {}
symat::symat(const std::string & symat_in) {
    std::ifstream ifs; ifs.open(symat_in);
        std::string line;
        // number of electronic states
        std::getline(ifs, line);
        std::getline(ifs, line);
        NStates_ = std::stoul(line);
        // symmetry (irreducible) of matrix elements
        std::getline(ifs, line);
        symmetry_ = CL::utility::matrix<size_t>(NStates_);
        for (size_t i = 0; i < NStates_; i++) {
            std::getline(ifs, line);
            auto strs = CL::utility::split(line);
            for (size_t j = 0; j < NStates_; j++)
            symmetry_[i][j] = std::stoul(strs[j]) - 1;
        }
        // elementary network
        std::getline(ifs, line);
        for (size_t i = 0; i < NStates_; i++)
        for (size_t j = i; j < NStates_; j++) {
            std::getline(ifs, line);
            auto strs = CL::utility::split(line);
            std::vector<size_t> dimensions(strs.size());
            for (size_t k = 0; k < strs.size(); k++)
            dimensions[k] = std::stoul(strs[k]);
            elements->push_back(scalar(dimensions, symmetry_[i][j] == 0));
        }
    ifs.close();
}
symat::~symat() {}

int64_t symat::NStates() const {return NStates_;}
CL::utility::matrix<size_t> symat::symmetry() const {return symmetry_;}

void symat::to(const torch::Dtype & dtype) {
    this->torch::nn::Module::to(dtype);
    elements->to(dtype);
}
CL::utility::matrix<std::vector<at::Tensor>> symat::parameters() {
    CL::utility::matrix<std::vector<at::Tensor>> ps(NStates_);
    size_t count = 0;
    for (int64_t i = 0; i < NStates_; i++)
    for (int64_t j = i; j < NStates_; j++) {
        ps[i][j] = elements[count]->parameters();
        count++;
    }
    return ps;
}

at::Tensor symat::forward(const CL::utility::matrix<at::Tensor> & xs) {
    assert(("Elements in xs must be vectors", xs[0][0].sizes().size() == 1));
    at::Tensor y = xs[0][0].new_empty({NStates_, NStates_});
    size_t count = 0;
    for (int64_t i = 0; i < NStates_; i++)
    for (int64_t j = i; j < NStates_; j++) {
        y[i][j] = elements[count]->as<scalar>()->operator()(xs[i][j]);
        count++;
    }
    return y;
}
at::Tensor symat::operator()(const CL::utility::matrix<at::Tensor> & xs) {return forward(xs);}

} // namespace obnet