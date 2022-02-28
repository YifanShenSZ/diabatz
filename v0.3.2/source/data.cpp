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
        auto reg = std::make_shared<RegHam>(stdreg, int2input);
        int thread = omp_get_thread_num();
        for (size_t j = 0; j < nduplicates; j++) pregss[thread].push_back(reg);
    }
    #pragma omp parallel for
    for (size_t i = 0; i < stddegset->size_int(); i++) {
        auto stddeg = stddegset->get(i);
        stddeg->subtract_ZeroPoint(zero_point);
        // duplicate data points based on weight
        size_t nduplicates = ceil(stddeg->weight());
        stddeg->set_weight(1.0);
        // precompute the input layers
        auto deg = std::make_shared<DegHam>(stddeg, int2input);
        int thread = omp_get_thread_num();
        for (size_t j = 0; j < nduplicates; j++) pdegss[thread].push_back(deg);
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

// given a regular data set
// return a shift and a width for feature scaling
std::tuple<CL::utility::matrix<at::Tensor>, CL::utility::matrix<at::Tensor>>
statisticize_regset(const std::shared_ptr<abinitio::DataSet<RegHam>> & regset) {
    size_t NExamples = regset->size_int(),
           NStates = Hdnet->NStates();
    // shift = input layer average
    // width = sqrt(input layer gradient metric maximum)
    CL::utility::matrix<at::Tensor> x_avg(NStates), s_max(NStates);
    // accumulate 0th example
    const auto & example = regset->examples()[0];
    const auto & xs = example->xs();
    const auto & JxrTs = example->JxrTs();
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = i; j < NStates; j++) {
        // input layer statistics
        const at::Tensor & x = xs[i][j];
        x_avg[i][j] = x.clone();
        // input layer gradient metric statistics
        const at::Tensor & JT = JxrTs[i][j];
        at::Tensor S = JT.transpose(0, 1).mm(JT).diag();
        s_max[i][j] = S.clone();
    }
    // accumulate remaining examples
    for (size_t iexample = 1; iexample < NExamples; iexample++) {
        const auto & example = regset->examples()[iexample];
        const auto & xs = example->xs();
        const auto & JxrTs = example->JxrTs();
        for (size_t i = 0; i < NStates; i++)
        for (size_t j = i; j < NStates; j++) {
            // input layer statistics
            const at::Tensor & x = xs[i][j];
            x_avg[i][j] += x;
            // input layer gradient metric statistics
            const at::Tensor & JT = JxrTs[i][j];
            at::Tensor S = JT.transpose(0, 1).mm(JT).diag();
            for (size_t k = 0; k < S.size(0); k++) {
                s_max[i][j][k].fill_(std::max(S[k].item<double>(), s_max[i][j][k].item<double>()));
            }
        }
    }
    // final process
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = i; j < NStates; j++) {
        // input layer average
        x_avg[i][j] /= (double)NExamples;
        // For asymmetric irreducibles, the average is always 0 because
        // the network design has considered the symmetry:
        // * On one hand, this has effectively performed a data augmentation
        //   that considers all symmetry-equivalent geometries,
        //   and the asymmetric average of all those geometires is always 0
        //   This would not affect standard deviation, since (input layer)^2 is symmetric
        // * On the other hand, this excludes bias from asymmetric network,
        //   so it is impossible for the network to output a same value after x -= avg
        if (Hdnet->irreds()[i][j] != 0) x_avg[i][j].fill_(0.0);
        // sqrt(input layer gradient metric maximum)
        s_max[i][j].sqrt_();
    }
    return std::make_tuple(x_avg, s_max);
}