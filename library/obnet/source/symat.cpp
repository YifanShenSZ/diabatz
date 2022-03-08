#include <CppLibrary/utility.hpp>

#include <obnet/scalar.hpp>

#include <obnet/symat.hpp>

namespace obnet {

symat::symat() {}
// This copy constructor performs a somewhat deepcopy,
// where new modules are generated and have same values as `source`
symat::symat(const std::shared_ptr<symat> & source) {
    this->NStates_ = source->NStates_;
    this-> irreds_ = source-> irreds_;
    for (size_t i = 0; i < source->elements->size(); i++)
    this->elements->push_back(scalar(source->elements[i]->as<scalar>()));
}
symat::symat(const std::string & symat_in) {
    std::ifstream ifs; ifs.open(symat_in);
    if (! ifs.good()) throw CL::utility::file_error(symat_in);
        std::string line;
        // number of electronic states
        std::getline(ifs, line);
        if (! ifs.good()) throw CL::utility::file_error(symat_in);
        std::getline(ifs, line);
        if (! ifs.good()) throw CL::utility::file_error(symat_in);
        NStates_ = std::stoul(line);
        // CNPI group irreducible of each matrix element
        std::getline(ifs, line);
        if (! ifs.good()) throw CL::utility::file_error(symat_in);
        irreds_ = CL::utility::matrix<size_t>(NStates_);
        for (size_t i = 0; i < NStates_; i++) {
            std::getline(ifs, line);
            if (! ifs.good()) throw CL::utility::file_error(symat_in);
            auto strs = CL::utility::split(line);
            for (size_t j = 0; j < NStates_; j++)
            irreds_[i][j] = std::stoul(strs[j]) - 1;
        }
        // elementary network
        std::getline(ifs, line);
        if (! ifs.good()) throw CL::utility::file_error(symat_in);
        for (size_t i = 0; i < NStates_; i++)
        for (size_t j = i; j < NStates_; j++) {
            std::getline(ifs, line);
            if (! ifs.good()) throw CL::utility::file_error(symat_in);
            auto strs = CL::utility::split(line);
            std::vector<size_t> dimensions(strs.size());
            for (size_t k = 0; k < strs.size(); k++)
            dimensions[k] = std::stoul(strs[k]);
            elements->push_back(scalar(dimensions, irreds_[i][j] == 0));
        }
    ifs.close();
}
symat::~symat() {}

const int64_t & symat::NStates() const {return NStates_;}
const CL::utility::matrix<size_t> & symat::irreds() const {return irreds_;}

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

void symat::copy_(const std::shared_ptr<symat> & source) {
    const auto & this_pars = elements->parameters();
    const auto & src_pars = source->elements->parameters();
    if (this_pars.size() != src_pars.size()) throw std::invalid_argument(
    "obnet::symat::copy_: inconsistent size");
    for (size_t i = 0; i < this_pars.size(); i++) {
        torch::NoGradGuard no_grad;
        this_pars[i].copy_(src_pars[i]);
    }
}

void symat::freeze() {
    size_t count = 0;
    for (int64_t i = 0; i < NStates_; i++)
    for (int64_t j = i; j < NStates_; j++) {
        elements[count]->as<scalar>()->freeze();
        count++;
    }
}
void symat::freeze(const std::vector<size_t> & indices) {
    for (const size_t & index : indices)
    elements[index]->as<scalar>()->freeze();
}

at::Tensor symat::forward(const CL::utility::matrix<at::Tensor> & xs) {
    if (xs.size(0) != NStates_) throw std::invalid_argument(
    "obnet::symat::forward: xs must be an NStates_ x NStates_ matrix");
    if (xs.size(1) != NStates_) throw std::invalid_argument(
    "obnet::symat::forward: xs must be an NStates_ x NStates_ matrix");
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

// output hidden layer values before activation to `os`
void symat::diagnostic(const CL::utility::matrix<at::Tensor> & xs, std::ostream & os) {
    if (xs.size(0) != NStates_) throw std::invalid_argument(
    "obnet::symat::forward: xs must be an NStates_ x NStates_ matrix");
    if (xs.size(1) != NStates_) throw std::invalid_argument(
    "obnet::symat::forward: xs must be an NStates_ x NStates_ matrix");
    at::Tensor y = xs[0][0].new_empty({NStates_, NStates_});
    size_t count = 0;
    for (int64_t i = 0; i < NStates_; i++)
    for (int64_t j = i; j < NStates_; j++) {
        os << "Matrix row " << i + 1 << " column " << j + 1 << ":\n";
        elements[count]->as<scalar>()->diagnostic(xs[i][j], os);
        os << '\n';
        count++;
    }
}

} // namespace obnet