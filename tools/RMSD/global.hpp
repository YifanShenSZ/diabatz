#ifndef global_hpp
#define global_hpp

#include <abinitio/Hamiltonian.hpp>
#include <abinitio/DataSet.hpp>

#include <Hd/Kernel.hpp>

extern std::shared_ptr<Hd::Kernel> HdKernel;

extern std::vector<std::shared_ptr<abinitio::RegHam>> regset;
extern std::vector<std::shared_ptr<abinitio::DegHam>> degset;

#endif