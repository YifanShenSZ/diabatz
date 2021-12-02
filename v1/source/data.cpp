#include <abinitio/SAreader.hpp>

#include "../include/global.hpp"
#include "../include/data.hpp"

std::tuple<std::shared_ptr<abinitio::DataSet<RegHam>>, std::shared_ptr<abinitio::DataSet<DegHam>>>
read_data(const std::vector<std::string> & user_list) {
    abinitio::SAReader reader(user_list, cart2CNPI);
    reader.pretty_print(std::cout);
    // read the data set in symmetry adapted internal coordinate in standard form
    std::shared_ptr<abinitio::DataSet<abinitio::RegSAHam>> stdregset;
    std::shared_ptr<abinitio::DataSet<abinitio::DegSAHam>> stddegset;
    std::tie(stdregset, stddegset) = reader.read_SAHamSet();
    // process data
    std::vector<std::shared_ptr<RegHam>> pregs;
    for (size_t i = 0; i < pregs.size(); i++) {
        auto stdreg = stdregset->get(i);
        // duplicate data points based on weight
        size_t nduplicates = ceil(stdreg->weight());
        stdreg->set_weight(1.0);
        // precompute the input layers
        auto reg = std::make_shared<RegHam>(stdreg, int2input);
        for (size_t j = 0; j < nduplicates; j++) pregs.push_back(reg);
    }
    std::vector<std::shared_ptr<DegHam>> pdegs;
    for (size_t i = 0; i < pdegs.size(); i++) {
        auto stddeg = stddegset->get(i);
        // duplicate data points based on weight
        size_t nduplicates = ceil(stddeg->weight());
        stddeg->set_weight(1.0);
        // precompute the input layers
        auto deg = std::make_shared<DegHam>(stddeg, int2input);
        for (size_t j = 0; j < nduplicates; j++) pdegs.push_back(deg);
    }
    // return
    std::shared_ptr<abinitio::DataSet<RegHam>> regset = std::make_shared<abinitio::DataSet<RegHam>>(pregs);
    std::shared_ptr<abinitio::DataSet<DegHam>> degset = std::make_shared<abinitio::DataSet<DegHam>>(pdegs);
    return std::make_tuple(regset, degset);
}

std::shared_ptr<abinitio::DataSet<Energy>> read_energy(const std::vector<std::string> & user_list) {
    abinitio::SAReader reader(user_list, cart2CNPI);
    reader.pretty_print(std::cout);
    // read the data set in symmetry adapted internal coordinate in standard form
    auto stdset = reader.read_SAEnergySet();
    // process data
    std::vector<std::shared_ptr<Energy>> penergies;
    for (size_t i = 0; i < stdset->size_int(); i++) {
        auto stdenergy = stdset->get(i);
        // duplicate data points based on weight
        size_t nduplicates = ceil(stdenergy->weight());
        stdenergy->set_weight(1.0);
        // precompute the input layers
        auto energy = std::make_shared<Energy>(stdenergy, int2input);
        for (size_t j = 0; j < nduplicates; j++) penergies.push_back(energy);
    }
    // return
    return std::make_shared<abinitio::DataSet<Energy>>(penergies);
}