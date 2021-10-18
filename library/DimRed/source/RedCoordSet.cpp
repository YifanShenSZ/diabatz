#include <CppLibrary/utility.hpp>

#include <DimRed/RedCoordSet.hpp>

namespace DimRed {

// Construct private sizes based on constructed networks `irreducibles`
void RedCoordSet::construct_sizes_() {
    TotalIndim_ = 0;
    TotalReddim_ = 0;
    TotalNPars_ = 0;
    indims_.resize(irreducibles->size());
    reddims_.resize(irreducibles->size());
    NPars_.resize(irreducibles->size());
    for (size_t i = 0; i < irreducibles->size(); i++) {
        auto & fcs = irreducibles[i]->as<Encoder>()->fcs;
        indims_[i] += fcs[0]->as<torch::nn::Linear>()->options.in_features();
        reddims_[i] += fcs[fcs->size() - 1]->as<torch::nn::Linear>()->options.out_features();
        NPars_[i] = 0;
        for (const at::Tensor & c : irreducibles[i]->parameters()) NPars_[i] += c.numel();
        TotalIndim_ += indims_[i];
        TotalReddim_ += reddims_[i];
        TotalNPars_ += NPars_[i];
    }
}

RedCoordSet::RedCoordSet() {}
// This copy constructor performs a somewhat deepcopy,
// where new modules are generated and have same values as `source`
RedCoordSet::RedCoordSet(const std::shared_ptr<RedCoordSet> & source) {
    torch::NoGradGuard no_grad;
    for (size_t i = 0; i < source->NIrreds(); i++)
    irreducibles->push_back(Encoder(source->irreducibles[i]->as<Encoder>()));
    this->construct_sizes_();
}
RedCoordSet::RedCoordSet(const std::string & redcoordset_in) {
    std::ifstream ifs; ifs.open(redcoordset_in);
    if (ifs.good()) {
        std::string line;
        // The 1st (totally symmetric) irreducible
        std::getline(ifs, line);
        if (! ifs.good()) throw CL::utility::file_error(redcoordset_in);
        auto strs = CL::utility::split(line);
        std::vector<size_t> dims(strs.size());
        for (size_t i = 0; i < strs.size(); i++) dims[i] = std::stoul(strs[i]);
        irreducibles->push_back(Encoder(dims, true));
        // The other irreducibles
        while (true) {
            std::getline(ifs, line);
            if (! ifs.good()) break;
            auto strs = CL::utility::split(line);
            std::vector<size_t> dims(strs.size());
            for (size_t i = 0; i < strs.size(); i++) dims[i] = std::stoul(strs[i]);
            irreducibles->push_back(Encoder(dims, false));
        }
    }
    ifs.close();
    this->construct_sizes_();
}
RedCoordSet::~RedCoordSet() {}

size_t RedCoordSet::NIrreds() const {return irreducibles->size();}

at::Tensor RedCoordSet::cat_Jrx(const std::vector<at::Tensor> & Jrxs) const {
    if (Jrxs.size() != irreducibles->size()) throw std::invalid_argument(
    "DimRed::RedCoordSet::cat_Jrx: inconsistent number of irreducible representations");
    at::Tensor Jrx = Jrxs[0].new_zeros({TotalReddim_, TotalIndim_});
    int64_t start_row = 0, start_col = 0;
    for (size_t i = 0; i < Jrxs.size(); i++) {
        if (Jrxs[i].size(0) != reddims_[i]) throw std::invalid_argument(
        "DimRed::RedCoordSet::cat_Jrx: inconsistent reduced coordinate dimension");
        if (Jrxs[i].size(1) != indims_[i]) throw std::invalid_argument(
        "DimRed::RedCoordSet::cat_Jrx: inconsistent input coordinate dimension");
        int64_t stop_row = start_row + reddims_[i],
                stop_col = start_col +  indims_[i];
        Jrx.slice(0, start_row, stop_row).slice(1, start_col, stop_col).copy_(Jrxs[i]);
        start_row = stop_row;
        start_col = stop_col;
    }
    return Jrx;
}
at::Tensor RedCoordSet::cat_Jrc(const std::vector<at::Tensor> & Jrcs) const {
    if (Jrcs.size() != irreducibles->size()) throw std::invalid_argument(
    "DimRed::RedCoordSet::cat_Jrc: inconsistent number of irreducible representations");
    at::Tensor Jrc = Jrcs[0].new_zeros({TotalReddim_, TotalNPars_});
    int64_t start_row = 0, start_col = 0;
    for (size_t i = 0; i < Jrcs.size(); i++) {
        if (Jrcs[i].size(0) != reddims_[i]) throw std::invalid_argument(
        "DimRed::RedCoordSet::cat_Jrc: inconsistent reduced coordinate dimension");
        if (Jrcs[i].size(1) != NPars_[i]) throw std::invalid_argument(
        "DimRed::RedCoordSet::cat_Jrc: inconsistent number of parameters");
        int64_t stop_row = start_row + reddims_[i],
                stop_col = start_col +   NPars_[i];
        Jrc.slice(0, start_row, stop_row).slice(1, start_col, stop_col).copy_(Jrcs[i]);
        start_row = stop_row;
        start_col = stop_col;
    }
    return Jrc;
}
at::Tensor RedCoordSet::cat_Krxc(const std::vector<at::Tensor> & Krxcs) const {
    if (Krxcs.size() != irreducibles->size()) throw std::invalid_argument(
    "DimRed::RedCoordSet::cat_Krxc: inconsistent number of irreducible representations");
    at::Tensor Krxc = Krxcs[0].new_zeros({TotalReddim_, TotalIndim_, TotalNPars_});
    int64_t start0 = 0, start1 = 0, start2 = 0;
    for (size_t i = 0; i < Krxcs.size(); i++) {
        if (Krxcs[i].size(0) != reddims_[i]) throw std::invalid_argument(
        "DimRed::RedCoordSet::cat_Krxc: inconsistent reduced coordinate dimension");
        if (Krxcs[i].size(1) != indims_[i]) throw std::invalid_argument(
        "DimRed::RedCoordSet::cat_Krxc: inconsistent input coordinate dimension");
        if (Krxcs[i].size(2) != NPars_[i]) throw std::invalid_argument(
        "DimRed::RedCoordSet::cat_Krxc: inconsistent number of parameters");
        int64_t stop0 = start0 + reddims_[i],
                stop1 = start1 +  indims_[i],
                stop2 = start2 +   NPars_[i];
        Krxc.slice(0, start0, stop0).slice(1, start1, stop1).slice(2, start2, stop2).copy_(Krxcs[i]);
        start0 = stop0;
        start1 = stop1;
        start2 = stop2;
    }
    return Krxc;
}

std::vector<at::Tensor> RedCoordSet::forward(const std::vector<at::Tensor> & xs) {
    if (xs.size() != irreducibles->size()) throw std::invalid_argument(
    "DimRed::RedCoordSet::forward: inconsistent number of irreducible representations");
    std::vector<at::Tensor> rs(irreducibles->size());
    for (size_t i = 0; i < irreducibles->size(); i++) {
        if (xs[i].sizes().size() != 1) throw std::invalid_argument(
        "DimRed::RedCoordSet::forward: input must be a vector");
        if (xs[i].size(0) != indims_[i]) throw std::invalid_argument(
        "DimRed::RedCoordSet::forward: inconsistent input coordinate dimension");
        rs[i] = irreducibles[i]->as<Encoder>()->operator()(xs[i]);
    }
    return rs;
}
std::vector<at::Tensor> RedCoordSet::operator()(const std::vector<at::Tensor> & xs) {return this->forward(xs);}

std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>>
RedCoordSet::compute_r_Jrx(const std::vector<at::Tensor> & xs) {
    if (xs.size() != irreducibles->size()) throw std::invalid_argument(
    "DimRed::RedCoordSet::compute_r_JrxT: inconsistent number of irreducible representations");
    std::vector<at::Tensor> rs(irreducibles->size()), Jrxs(irreducibles->size());
    for (size_t i = 0; i < irreducibles->size(); i++) {
        if (xs[i].sizes().size() != 1) throw std::invalid_argument(
        "DimRed::RedCoordSet::forward: input must be a vector");
        if (xs[i].size(0) != indims_[i]) throw std::invalid_argument(
        "DimRed::RedCoordSet::forward: inconsistent input coordinate dimension");
        // r
        rs[i] = irreducibles[i]->as<Encoder>()->operator()(xs[i]);
        // Jrx
        Jrxs[i] = xs[i].new_empty({rs[i].size(0), xs[i].size(0)});
        for (int64_t j = 0; j < rs[i].size(0); j++) {
            auto g = torch::autograd::grad({rs[i][j]}, {xs[i]}, {}, true);
            Jrxs[i][j].copy_(g[0]);
        }
    }
    return std::make_tuple(rs, Jrxs);
}

std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>, std::vector<at::Tensor>, std::vector<at::Tensor>>
RedCoordSet::compute_r_Jrx_Jrc_Krxc(const std::vector<at::Tensor> & xs) {
    if (xs.size() != irreducibles->size()) throw std::invalid_argument(
    "DimRed::RedCoordSet::compute_r_JrxT: inconsistent number of irreducible representations");
    std::vector<at::Tensor> rs(irreducibles->size()),
                            Jrxs(irreducibles->size()), Jrcs(irreducibles->size()),
                            Krxcs(irreducibles->size());
    for (size_t i = 0; i < irreducibles->size(); i++) {
        if (xs[i].sizes().size() != 1) throw std::invalid_argument(
        "DimRed::RedCoordSet::forward: input must be a vector");
        if (xs[i].size(0) != indims_[i]) throw std::invalid_argument(
        "DimRed::RedCoordSet::forward: inconsistent input coordinate dimension");
        // r
        rs[i] = irreducibles[i]->as<Encoder>()->operator()(xs[i]);
        // Jrx
        Jrxs[i] = xs[i].new_empty({rs[i].size(0), xs[i].size(0)});
        for (int64_t j = 0; j < rs[i].size(0); j++) {
            auto g = torch::autograd::grad({rs[i][j]}, {xs[i]}, {}, true, true);
            Jrxs[i][j] = g[0];
        }
        // Jrc
        const auto & cs = irreducibles[i]->parameters();
        Jrcs[i] = xs[i].new_empty({rs[i].size(0), NPars_[i]});
        for (int64_t j = 0; j < rs[i].size(0); j++) {
            auto gs = torch::autograd::grad({rs[i][j]}, cs, {}, true);
            for (at::Tensor & g : gs) if (g.sizes().size() != 1) g = g.view(g.numel());
            Jrcs[i][j].copy_(at::cat(gs));
        }
        // Krxc
        Krxcs[i] = xs[i].new_empty({rs[i].size(0), xs[i].size(0), NPars_[i]});
        for (int64_t j = 0; j < rs[i].size(0); j++)
        for (int64_t k = 0; k < xs[i].size(0); k++) {
            auto gs = torch::autograd::grad({Jrxs[i][j][k]}, cs, {}, true);
            for (at::Tensor & g : gs) if (g.sizes().size() != 1) g = g.view(g.numel());
            Krxcs[i][j][k].copy_(at::cat(gs));
        }
    }
    return std::make_tuple(rs, Jrxs, Jrcs, Krxcs);
}

} // namespace DimRed