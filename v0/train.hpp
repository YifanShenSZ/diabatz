#ifndef train_hpp
#define train_hpp

#include <tchem/chemistry.hpp>

#include <obnet/symat.hpp>

#include "data.hpp"

namespace train {

extern size_t NStates;

// data set
extern std::vector<std::shared_ptr<RegHam>> regset;
extern std::vector<std::shared_ptr<DegHam>> degset;

// the "unit" of energy, accounting for the unit difference between energy and gradient
extern double unit, unit_square;

extern std::vector<std::shared_ptr<tchem::chem::Phaser>> phasers;

extern int32_t OMP_NUM_THREADS;
// Each thread owns a copy of Hd network
// Thread 0 shares the original Hdnet
extern std::vector<std::shared_ptr<obnet::symat>> Hdnets;
// Each thread owns a chunk of data, works on a segment of residue or Jacobian
// Thread i owns regular data [regchunk[i] - regchunk[0], regchunk[i]),
//            degenerate data [degchunk[i] - degchunk[0], degchunk[i]),
// works on rows [segstart[i], segstart[i + 1]) 
extern std::vector<size_t> regchunk, degchunk, segstart;

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

void init_train();

} // namespace train

#endif