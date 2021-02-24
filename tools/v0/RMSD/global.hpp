#ifndef global_hpp
#define global_hpp

#include <abinitio/Hamiltonian.hpp>
#include <abinitio/DataSet.hpp>

#include <Hd/kernel.hpp>

extern std::shared_ptr<Hd::kernel> Hdkernel;

extern std::vector<std::shared_ptr<abinitio::RegHam>> regset;
extern std::vector<std::shared_ptr<abinitio::DegHam>> degset;

#endif