#ifndef train_hpp
#define train_hpp

#include <tchem/linalg.hpp>
#include <tchem/phaser.hpp>

#include <obnet/symat.hpp>

#include "data.hpp"

namespace train {

extern int64_t NStates;

extern std::vector<std::shared_ptr<tchem::Phaser>> phasers;

// data set
extern std::vector<std::shared_ptr<RegHam>> regset;
extern std::vector<std::shared_ptr<DegHam>> degset;

// the "unit" of energy, accounting for the unit difference between energy and gradient
extern double unit, unit_square;

// Number of least square equations and fitting parameters
extern int32_t NEqs, NPars;

extern size_t OMP_NUM_THREADS;
// Each thread owns a copy of Hd network
// Thread 0 shares the original Hdnet
extern std::vector<std::shared_ptr<obnet::symat>> Hdnets;
// Each thread owns a chunk of data
extern std::vector<std::vector<std::shared_ptr<RegHam>>> regchunk;
extern std::vector<std::vector<std::shared_ptr<DegHam>>> degchunk;
// Each thread works on a segment of residue or Jacobian
// Thread i works on rows [segstart[i], segstart[i + 1])
extern std::vector<size_t> segstart;

inline void p2c(const int32_t & thread, double * c) {
    size_t count = 0;
    for (const at::Tensor & p : Hdnets[thread]->elements->parameters()) {
        size_t numel = p.numel();
        std::memcpy(&(c[count]), p.data_ptr<double>(), numel * sizeof(double));
        count += numel;
    }
}
inline void c2p(const double * c, const int32_t & thread) {
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
    DrHa = DrHa.slice(0, 0, NStates_data).slice(1, 0, NStates_data);
    size_t iphase = phasers[NStates_data]->iphase_min(DrHa, DrHa_data);
    at::Tensor states_view = states.slice(1, 0, NStates_data);
    phasers[NStates_data]->alter_states_(states_view, iphase);
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
    size_t iphase = phasers[NStates]->iphase_min(Hc, DrHc, Hc_data, DrHc_data, unit_square);
    phasers[NStates]->alter_states_(eigvec, iphase);
    return std::make_tuple(eigval, eigvec);
}

} // namespace train

void initialize(
const std::shared_ptr<abinitio::DataSet<RegHam>> & regset,
const std::shared_ptr<abinitio::DataSet<DegHam>> & degset);

void optimize(const bool & regularized, const size_t & max_iteration);

#endif