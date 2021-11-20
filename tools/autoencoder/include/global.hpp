#ifndef global_hpp
#define global_hpp

#include <SASDIC/SASDICSet.hpp>

#include <DimRed/encoder.hpp>
#include <DimRed/decoder.hpp>

#include <abinitio/DataSet.hpp>
#include <abinitio/SAgeometry.hpp>

extern std::shared_ptr<SASDIC::SASDICSet> sasicset;

// Given Cartesian coordinate r,
// return CNPI group symmetry adapted internal coordinates and corresponding Jacobians
std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> cart2int(const at::Tensor & r);

extern size_t irreducible;
extern std::shared_ptr<DimRed::Encoder> encoder;
extern std::shared_ptr<DimRed::Decoder> decoder;

extern std::shared_ptr<abinitio::DataSet<abinitio::SAGeometry>> geom_set;

std::vector<size_t> read_vector(const std::string & file);

double RMSD();

#endif