#include <abinitio/SAreader.hpp>

#include "../include/global.hpp"
#include "../include/data.hpp"

std::tuple<std::shared_ptr<abinitio::DataSet<RegHam>>, std::shared_ptr<abinitio::DataSet<DegHam>>>
read_data(const std::vector<std::string> & user_list, const double & zero_point) {
    abinitio::SAReader reader(user_list, cart2CNPI);
    reader.pretty_print(std::cout);
    // read the data set in symmetry adapted internal coordinate in standard form
    std::shared_ptr<abinitio::DataSet<abinitio::RegSAHam>> stdregset;
    std::shared_ptr<abinitio::DataSet<abinitio::DegSAHam>> stddegset;
    std::tie(stdregset, stddegset) = reader.read_SAHamSet();
    // process data
    size_t OMP_NUM_THREADS = omp_get_max_threads();
    std::vector<std::vector<std::shared_ptr<RegHam>>> pregss(OMP_NUM_THREADS);
    std::vector<std::vector<std::shared_ptr<DegHam>>> pdegss(OMP_NUM_THREADS);
    #pragma omp parallel for
    for (size_t i = 0; i < stdregset->size_int(); i++) {
        auto stdreg = stdregset->get(i);
        stdreg->subtract_ZeroPoint(zero_point);
        // duplicate data points based on weight
        size_t nduplicates = ceil(stdreg->weight());
        stdreg->set_weight(1.0);
        // precompute the input layers
        int thread = omp_get_thread_num();
        for (size_t j = 0; j < nduplicates; j++) pregss[thread].push_back(std::make_shared<RegHam>(stdreg, int2input));
    }
    #pragma omp parallel for
    for (size_t i = 0; i < stddegset->size_int(); i++) {
        auto stddeg = stddegset->get(i);
        stddeg->subtract_ZeroPoint(zero_point);
        // duplicate data points based on weight
        size_t nduplicates = ceil(stddeg->weight());
        stddeg->set_weight(1.0);
        // precompute the input layers
        int thread = omp_get_thread_num();
        for (size_t j = 0; j < nduplicates; j++) pdegss[thread].push_back(std::make_shared<DegHam>(stddeg, int2input));
    }
    size_t nregs = 0, ndegs = 0;
    for (size_t i = 0; i < OMP_NUM_THREADS; i++) {
        nregs += pregss[i].size();
        ndegs += pdegss[i].size();
    }
    std::vector<std::shared_ptr<RegHam>> pregs;
    std::vector<std::shared_ptr<DegHam>> pdegs;
    pregs.reserve(nregs);
    pdegs.reserve(ndegs);
    for (size_t i = 0; i < OMP_NUM_THREADS; i++) {
        const auto & pregss_i = pregss[i];
        const auto & pdegss_i = pdegss[i];
        pregs.insert(pregs.end(), pregss_i.begin(), pregss_i.end());
        pdegs.insert(pdegs.end(), pdegss_i.begin(), pdegss_i.end());
    }
    // return
    std::shared_ptr<abinitio::DataSet<RegHam>> regset = std::make_shared<abinitio::DataSet<RegHam>>(pregs);
    std::shared_ptr<abinitio::DataSet<DegHam>> degset = std::make_shared<abinitio::DataSet<DegHam>>(pdegs);
    return std::make_tuple(regset, degset);
}

std::shared_ptr<abinitio::DataSet<Energy>> read_energy(const std::vector<std::string> & user_list, const double & zero_point) {
    abinitio::SAReader reader(user_list, cart2CNPI);
    reader.pretty_print(std::cout);
    // read the data set in symmetry adapted internal coordinate in standard form
    auto stdset = reader.read_SAEnergySet();
    // process data
    std::vector<std::shared_ptr<Energy>> penergies;
    for (size_t i = 0; i < stdset->size_int(); i++) {
        auto stdenergy = stdset->get(i);
        stdenergy->subtract_ZeroPoint(zero_point);
        // duplicate data points based on weight
        size_t nduplicates = ceil(stdenergy->weight());
        stdenergy->set_weight(1.0);
        // precompute the input layers
        for (size_t j = 0; j < nduplicates; j++) penergies.push_back(std::make_shared<Energy>(stdenergy, int2input));
    }
    // return
    return std::make_shared<abinitio::DataSet<Energy>>(penergies);
}