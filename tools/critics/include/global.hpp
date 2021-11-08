#ifndef global_hpp
#define global_hpp

#include <tchem/intcoord.hpp>

#include <Hd/kernel.hpp>

#include "fixed_intcoord.hpp"

extern std::shared_ptr<tchem::IC::IntCoordSet> intcoordset;

extern size_t target_state;

extern std::shared_ptr<Hd::kernel> Hdkernel;

extern std::shared_ptr<Fixed_intcoord> fixed_intcoord;

at::Tensor int2cart(const at::Tensor & q, const at::Tensor & init_guess,
const std::shared_ptr<tchem::IC::IntCoordSet> & _intcoordset);

#endif