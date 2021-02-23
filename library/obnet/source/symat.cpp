#include <CppLibrary/utility.hpp>

#include <obnet/scalar.hpp>

#include <obnet/symat.hpp>

namespace obnet {

symat::symat() {}
// This copy constructor performs a somewhat deepcopy,
// where new modules are generated and have same values as `source`
// We do not use const reference because
// torch::nn::ModuleList::operator[] does not support `const`,
// although this constructor would not change `source` of course
symat::symat(const std::shared_ptr<symat> & source) {
    this->NStates_ = source->NStates_;
    this-> irreds_ = source-> irreds_;
    for (size_t i = 0; i < source->elements->size(); i++)
    this->elements->push_back(scalar(source->elements[i]->as<scalar>()));
}
symat::symat(const std::string & symat_in) {
    std::ifstream ifs; ifs.open(symat_in);
        std::string line;
        // number of electronic states
        std::getline(ifs, line);
        assert(("End of file during reading " + symat_in, ifs.good()));
        std::getline(ifs, line);
        assert(("End of file during reading " + symat_in, ifs.good()));
        NStates_ = std::stoul(line);
        // CNPI group irreducible of each matrix element
        std::getline(ifs, line);
        assert(("End of file during reading " + symat_in, ifs.good()));
        irreds_ = CL::utility::matrix<size_t>(NStates_);
        for (size_t i = 0; i < NStates_; i++) {
            std::getline(ifs, line);
            assert(("End of file during reading " + symat_in, ifs.good()));
            auto strs = CL::utility::split(line);
            for (size_t j = 0; j < NStates_; j++)
            irreds_[i][j] = std::stoul(strs[j]) - 1;
        }
        // elementary network
        std::getline(ifs, line);
        assert(("End of file during reading " + symat_in, ifs.good()));
        for (size_t i = 0; i < NStates_; i++)
        for (size_t j = i; j < NStates_; j++) {
            std::getline(ifs, line);
            assert(("End of file during reading " + symat_in, ifs.good()));
            auto strs = CL::utility::split(line);
            std::vector<size_t> dimensions(strs.size());
            for (size_t k = 0; k < strs.size(); k++)
            dimensions[k] = std::stoul(strs[k]);
            elements->push_back(scalar(dimensions, irreds_[i][j] == 0));
        }
    ifs.close();
}
symat::~symat() {}

int64_t symat::NStates() const {return NStates_;}
CL::utility::matrix<size_t> symat::irreds() const {return irreds_;}

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