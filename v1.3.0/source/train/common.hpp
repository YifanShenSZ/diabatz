#ifndef train_common_hpp
#define train_common_hpp

#include <tchem/linalg.hpp>
#include <tchem/chemistry.hpp>

#include "../../include/global.hpp"

namespace train {

extern int64_t NStates;

extern std::shared_ptr<tchem::chem::Orderer> orderer;

extern size_t OMP_NUM_THREADS;

// each thread owns a copy of Hd network
// thread 0 shares the original Hdnet
extern std::vector<std::shared_ptr<obnet::symat>> Hdnets;

inline void p2c(const size_t & thread, double * c) {
    size_t count = 0;
    for (const at::Tensor & p : Hdnets[thread]->elements->parameters()) {
        size_t numel = p.numel();
        std::memcpy(&(c[count]), p.data_ptr<double>(), numel * sizeof(double));
        count += numel;
    }
}
inline void c2p(const double * c, const size_t & thread) {
    torch::NoGradGuard no_grad;
    size_t count = 0;
    for (const at::Tensor & p : Hdnets[thread]->elements->parameters()) {
        size_t numel = p.numel();
        std::memcpy(p.data_ptr<double>(), &(c[count]), numel * sizeof(double));
        count += numel;
    }
}

inline std::tuple<at::Tensor, at::Tensor> define_adiabatz(
const at::Tensor & Hd, const at::Tensor & DqHd,
const at::Tensor & JqrT, const int64_t & cartdim,
const int64_t & NStates_data, const at::Tensor & DrHa_data) {
    at::Tensor energy, states;
    std::tie(energy, states) = Hd.symeig(true);
    at::Tensor DrHd = DqHd.new_empty({NStates, NStates, cartdim});
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = i; j < NStates; j++)
    DrHd[i][j] = JqrT.mv(DqHd[i][j]);
    at::Tensor DrHa = tchem::linalg::UT_sy_U(DrHd, states);
    size_t ipermutation, iphase;
    std::tie(ipermutation, iphase) = orderer->ipermutation_iphase_min(DrHa, DrHa_data);
    orderer->alter_eigvals_(energy, ipermutation);
    orderer->alter_states_(states, ipermutation, NStates_data, iphase);
    return std::make_tuple(energy, states);
}

inline std::tuple<at::Tensor, at::Tensor> define_composite(
const at::Tensor & Hd, const at::Tensor & DqHd,
const at::Tensor & JqrT, const int64_t & cartdim,
const at::Tensor & Hc_data, const at::Tensor & DrHc_data) {
    at::Tensor DrHd = DqHd.new_empty({NStates, NStates, cartdim});
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = i; j < NStates; j++)
    DrHd[i][j] = JqrT.mv(DqHd[i][j]);
    at::Tensor dHdH = tchem::linalg::sy3matdotmul(DrHd, DrHd);
    at::Tensor eigval, eigvec;
    std::tie(eigval, eigvec) = dHdH.symeig(true);
    at::Tensor   Hc = tchem::linalg::UT_sy_U(  Hd, eigvec),
               DrHc = tchem::linalg::UT_sy_U(DrHd, eigvec);
    size_t ipermutation, iphase;
    std::tie(ipermutation, iphase) = orderer->ipermutation_iphase_min(Hc, DrHc, Hc_data, DrHc_data, unit_square);
    orderer->alter_eigvals_(eigval, ipermutation);
    orderer->alter_states_(eigvec, ipermutation, NStates, iphase);
    return std::make_tuple(eigval, eigvec);
}

} // namespace train

#include "../../include/data_classes.hpp"

namespace train { namespace trust_region {

// data set
extern std::vector<std::shared_ptr<RegHam>> regset;
extern std::vector<std::shared_ptr<DegHam>> degset;
extern std::vector<std::shared_ptr<Energy>> energy_set;

// Each thread owns a chunk of data
extern std::vector<std::vector<std::shared_ptr<RegHam>>> regchunk;
extern std::vector<std::vector<std::shared_ptr<DegHam>>> degchunk;
extern std::vector<std::vector<std::shared_ptr<Energy>>> energy_chunk;

// Each thread works on a segment of residue or Jacobian
// Thread i works on rows [segstart[i], segstart[i + 1])
extern std::vector<size_t> segstart;

} // namespace trust_region
} // namespace train

#endif