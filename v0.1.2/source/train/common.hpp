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
    orderer->alter_states_(eigvec, ipermutation, NStates, iphase);
    return std::make_tuple(eigval, eigvec);
}

} // namespace train

#include "../../include/data.hpp"

namespace train { namespace torch_optim {

at::Tensor reg_residue(const std::vector<std::shared_ptr<RegHam>> & batch);

at::Tensor deg_residue(const std::vector<std::shared_ptr<DegHam>> & batch);

at::Tensor energy_residue(const std::vector<std::shared_ptr<Energy>> & batch);

at::Tensor reg_gradient(const std::vector<std::shared_ptr<RegHam>> & batch);

at::Tensor deg_gradient(const std::vector<std::shared_ptr<DegHam>> & batch);

at::Tensor energy_gradient(const std::vector<std::shared_ptr<Energy>> & batch);

} // namespace torch_optim
} // namespace train

#endif