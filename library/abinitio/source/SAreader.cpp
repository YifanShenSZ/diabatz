#include <CppLibrary/utility.hpp>

#include <tchem/chemistry.hpp>

#include <abinitio/SAreader.hpp>

namespace abinitio {

SAReader::SAReader() {}
// see the base class constructor for details of `user_list`
// `cart2CNPI` takes in Cartesian coordinate r, returns CNPI group symmetry adapted internal coordinates and their Jacobians over r
SAReader::SAReader(const std::vector<std::string> & user_list,
std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> (*_cart2CNPI)(const at::Tensor &),
const double& _deg_thresh)
: Reader(user_list, _deg_thresh), cart2CNPI_(_cart2CNPI) {}
SAReader::~SAReader() {}

// read geometries in symmetry adapted internal coordinates
std::shared_ptr<DataSet<SAGeometry>> SAReader::read_SAGeomSet() const {
    std::vector<std::shared_ptr<SAGeometry>> pgeoms;
    for (const std::string & data_directory : data_directories_) {
        std::vector<SAGeomLoader> loaders(NData(data_directory));
        size_t cartdim = 3 * NAtoms();
        for (auto & loader : loaders) loader.reset(cartdim);
        load_weight    (loaders, data_directory);
        load_geom      (loaders, data_directory);
        load_CNPI2point(loaders, data_directory);
        load_pointDefs (loaders, data_directory);
        for (const auto & loader : loaders) pgeoms.push_back(std::make_shared<SAGeometry>(loader, cart2CNPI_));
    }
    std::shared_ptr<DataSet<SAGeometry>> GeomSet = std::make_shared<DataSet<SAGeometry>>(pgeoms);
    return GeomSet;
}

// read energies in symmetry adapted internal coordinates
std::shared_ptr<DataSet<SAEnergy>> SAReader::read_SAEnergySet() const {
    std::vector<std::shared_ptr<SAEnergy>> penergies;
    for (const std::string & data_directory : data_directories_) {
        std::vector<SAEnergyLoader> loaders(NData(data_directory));
        size_t cartdim = 3 * NAtoms(),
               nstates = NStates(data_directory);
        for (auto & loader : loaders) loader.reset(cartdim, nstates);
        load_weight    (loaders, data_directory);
        load_geom      (loaders, data_directory);
        load_CNPI2point(loaders, data_directory);
        load_pointDefs (loaders, data_directory);
        load_energy    (loaders, data_directory);
        for (auto & loader : loaders) penergies.push_back(std::make_shared<SAEnergy>(loader, cart2CNPI_));
    }
    std::shared_ptr<DataSet<SAEnergy>> EnergySet = std::make_shared<DataSet<SAEnergy>>(penergies);
    return EnergySet;
}

// read Hamiltonians in symmetry adapted internal coordinates
std::tuple<std::shared_ptr<DataSet<RegSAHam>>, std::shared_ptr<DataSet<DegSAHam>>>
SAReader::read_SAHamSet() const {
    std::vector<std::shared_ptr<RegSAHam>> pregs;
    std::vector<std::shared_ptr<DegSAHam>> pdegs;
    for (const std::string & data_directory : data_directories_) {
        std::vector<SAHamLoader> loaders(NData(data_directory));
        size_t cartdim = 3 * NAtoms(),
               nstates = NStates(data_directory);
        for (auto & loader : loaders) loader.reset(cartdim, nstates);
        load_weight    (loaders, data_directory);
        load_geom      (loaders, data_directory);
        load_CNPI2point(loaders, data_directory);
        load_pointDefs (loaders, data_directory);
        load_energy    (loaders, data_directory);
        load_dH        (loaders, data_directory);
        for (auto & loader : loaders) {
            if (tchem::chem::check_degeneracy(loader.energy, deg_thresh_)) pdegs.push_back(std::make_shared<DegSAHam>(loader, cart2CNPI_));
            else                                                           pregs.push_back(std::make_shared<RegSAHam>(loader, cart2CNPI_));
        }
    }
    std::shared_ptr<DataSet<RegSAHam>> RegSet = std::make_shared<DataSet<RegSAHam>>(pregs);
    std::shared_ptr<DataSet<DegSAHam>> DegSet = std::make_shared<DataSet<DegSAHam>>(pdegs);
    return std::make_tuple(RegSet, DegSet);
}

} // namespace abinitio