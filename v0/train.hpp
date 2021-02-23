#ifndef train_hpp
#define train_hpp

#include <tchem/chemistry.hpp>

#include <obnet/symat.hpp>

#include "data.hpp"

namespace train {

extern size_t NStates;

extern std::vector<std::shared_ptr<tchem::chem::Phaser>> phasers;

// data set
extern std::vector<std::shared_ptr<RegHam>> regset;
extern std::vector<std::shared_ptr<DegHam>> degset;

// the "unit" of energy, accounting for the unit difference between energy and gradient
extern double unit;

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

} // namespace train

void initialize(
const std::shared_ptr<abinitio::DataSet<RegHam>> & regset,
const std::shared_ptr<abinitio::DataSet<DegHam>> & degset);

void optimize(const size_t & max_iteration);

#endif