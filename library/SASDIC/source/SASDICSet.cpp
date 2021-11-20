#include <regex>

#include <CppLibrary/utility.hpp>
#include <CppLibrary/chemistry.hpp>

#include <tchem/utility.hpp>

#include <SASDIC/SASDICSet.hpp>

namespace SASDIC {

SASDICSet::SASDICSet() {}
// internal coordinate definition format (Columbus7, default)
// internal coordinate definition file
// symmetry adaptation and scale definition file
SASDICSet::SASDICSet(const std::string & format, const std::string & IC_file, const std::string & SAS_file)
: tchem::IC::IntCoordSet(format, IC_file) {
    int64_t intdim = this->IntCoordSet::size();
    c10::TensorOptions top = at::TensorOptions().dtype(torch::kFloat64);
    std::ifstream ifs; ifs.open(SAS_file);
    if (! ifs.good()) throw CL::utility::file_error(SAS_file);
    else {
        std::string line;
        // internal coordinate origin
        std::string origin_file;
        std::getline(ifs, line);
        std::getline(ifs, origin_file);
        CL::utility::trim(origin_file);
        origin_ = tchem::utility::read_vector(origin_file);
        // internal coordinates to be scaled
        std::getline(ifs, line);
        while (true) {
            std::getline(ifs, line);
            std::vector<std::string> strs = CL::utility::split(line);
            if (! std::regex_match(strs[0], std::regex("\\d+"))) break;
            if (strs.size() < 3) throw std::invalid_argument(
            "SASDIC::SASDICSet::SASDICSet: wrong input format in " + SAS_file);
            size_t self = std::stoul(strs[0]) - 1;
            std::string scaling_function = strs[1];
            std::vector<double> parameters(strs.size() - 2);
            for (size_t i = 0; i < parameters.size(); i++) parameters[i] = std::stod(strs[2 + i]);
            scalers_.push_back(Scaler(self, scaling_function, parameters));
        }
        scaling_complete_ = at::eye(intdim, top);
        for (const Scaler & scaler : scalers_) {
            const size_t & self = scaler.self();
            scaling_complete_[self][self].fill_(0.0);
        }
        // symmetry adapted linear combinations of each irreducible
        while (ifs.good()) {
            sasdicss_.push_back(std::vector<SASDIC>());
            std::vector<SASDIC> & sasdics = sasdicss_.back();
            while (true) {
                std::getline(ifs, line);
                if (! ifs.good()) break;
                std::forward_list<std::string> strs;
                CL::utility::split(line, strs);
                if (! std::regex_match(strs.front(), std::regex("-?\\d+\\.?\\d*"))) break;
                if (std::regex_match(strs.front(), std::regex("\\d+"))) {
                    sasdics.push_back(SASDIC());
                    strs.pop_front();
                }
                double coeff = std::stod(strs.front()); strs.pop_front();
                size_t index = std::stoul(strs.front()) - 1;
                sasdics.back().append(coeff, index);
            }
            // normalize linear combination coefficients
            for (SASDIC & sasdic : sasdics) sasdic.normalize();
        }
    }
    ifs.close();
}
SASDICSet::~SASDICSet() {}

const at::Tensor & SASDICSet::origin() const {return origin_;}

// number of irreducible representations
size_t SASDICSet::NIrreds() const {return sasdicss_.size();}
// number of symmetry adapted and scaled dimensionless internal coordinates per irreducible
std::vector<size_t> SASDICSet::NSASDICs() const {
    std::vector<size_t> N(NIrreds());
    for (size_t i = 0; i < NIrreds(); i++) N[i] = sasdicss_[i].size();
    return N;
}
// number of internal coordinates
size_t SASDICSet::intdim() const {
    size_t intdim = 0;
    for (const auto & sasdics : sasdicss_) intdim += sasdics.size();
    return intdim;
}

// given internal coordinates q, return SASDIC
std::vector<at::Tensor> SASDICSet::operator()(const at::Tensor & q) const {
    if (q.sizes().size() != 1) throw std::invalid_argument(
    "SASDIC::SASDICSet::operator(): q must be a vector");
    if (q.size(0) != this->size()) throw std::invalid_argument(
    "SASDIC::SASDICSet::operator(): inconsisten dimension between q and this internal coordinate system");
    // nondimensionalize
    at::Tensor dics = q - origin_;
    for (size_t i = 0; i < this->size(); i++)
    if ((*this)[i][0].second.type() == "stretching")
    dics[i] = dics[i] / origin_[i];
    // scale
    at::Tensor sdics = scaling_complete_.mv(dics);
    for (const Scaler & scaler : scalers_) {
        const size_t & self = scaler.self();
        sdics[self] = scaler(dics[self]);
    }
    // symmetrize
    std::vector<at::Tensor> sasdicss(NIrreds());
    for (size_t i = 0; i < NIrreds(); i++) {
        int64_t intdim = sasdicss_[i].size();
        sasdicss[i] = q.new_zeros(intdim);
        for (size_t j = 0; j < intdim; j++) sasdicss[i][j] = sasdicss_[i][j](sdics);
    }
    return sasdicss;
}

} // namespace SASDIC