#include <CppLibrary/utility.hpp>

#include "../include/global.hpp"

std::shared_ptr<SASDIC::SASDICSet> sasicset;

// Given Cartesian coordinate r,
// return CNPI group symmetry adapted internal coordinates and corresponding Jacobians
std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> cart2int(const at::Tensor & r) {
    assert(("Define CNPI group symmetry adaptated and scaled internal coordinate before use", sasicset));
    // Cartesian coordinate -> internal coordinate
    at::Tensor q, J;
    std::tie(q, J) = sasicset->compute_IC_J(r);
    q.set_requires_grad(true);
    // internal coordinate -> CNPI group symmetry adapted internal coordinate
    std::vector<at::Tensor> qs = (*sasicset)(q);
    std::vector<at::Tensor> Js = std::vector<at::Tensor>(qs.size());
    for (size_t i = 0; i < qs.size(); i++) {
        Js[i] = qs[i].new_empty({qs[i].size(0), q.size(0)});
        for (size_t j = 0; j < qs[i].size(0); j++) {
            std::vector<at::Tensor> g = torch::autograd::grad({qs[i][j]}, {q}, {}, true);
            Js[i][j].copy_(g[0]);
        }
        Js[i] = Js[i].mm(J);
    }
    // Free autograd graph
    for (at::Tensor & q : qs) q.detach_();
    return std::make_tuple(qs, Js);
}

size_t irreducible;
std::shared_ptr<DimRed::Encoder> encoder;
std::shared_ptr<DimRed::Decoder> decoder;

std::shared_ptr<abinitio::DataSet<abinitio::SAGeometry>> geom_set;

std::vector<size_t> read_vector(const std::string & file) {
    std::vector<size_t> data;
    std::ifstream ifs; ifs.open(file);
    if (! ifs.good()) throw CL::utility::file_error(file);
    else {
        size_t reader;
        ifs >> reader;
        while (ifs.good()) {
            data.push_back(reader);
            ifs >> reader;
        }
    }
    ifs.close();
    return data;
}

double RMSD() {
    double rmsd = 0.0;
    for (const auto & data : geom_set->examples()) {
        torch::NoGradGuard no_grad;
        const auto & x = data->qs()[irreducible];
        rmsd += torch::mse_loss(decoder->forward(encoder->forward(x)), x, at::Reduction::Sum).item<double>();
    }
    return sqrt(rmsd / geom_set->size_int());
}