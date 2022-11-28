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
    std::vector<std::shared_ptr<RegHam>> pregs(stdregset->size_int());
    #pragma omp parallel for
    for (size_t i = 0; i < pregs.size(); i++) {
        auto reg = stdregset->get(i);
        // precompute the input layers
        pregs[i] = std::make_shared<RegHam>(reg, int2input1, int2input2);
    }
    std::vector<std::shared_ptr<DegHam>> pdegs(stddegset->size_int());
    #pragma omp parallel for
    for (size_t i = 0; i < pdegs.size(); i++) {
        auto deg = stddegset->get(i);
        // precompute the input layers
        pdegs[i] = std::make_shared<DegHam>(deg, int2input1, int2input2);
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
    std::vector<std::shared_ptr<Energy>> penergies(stdset->size_int());
    #pragma omp parallel for
    for (size_t i = 0; i < penergies.size(); i++) {
        auto energy = stdset->get(i);
        // precompute the input layers
        penergies[i] = std::make_shared<Energy>(energy, int2input1, int2input2);
    }
    // return
    return std::make_shared<abinitio::DataSet<Energy>>(penergies);
}

// given a regular data set
// return a shift and a width for feature scaling
std::tuple<CL::utility::matrix<at::Tensor>, CL::utility::matrix<at::Tensor>,
           CL::utility::matrix<at::Tensor>, CL::utility::matrix<at::Tensor>>
statisticize_regset(const std::shared_ptr<abinitio::DataSet<RegHam>> & regset) {
    size_t NExamples = regset->size_int(),
           NStates = Hdnet1->NStates();
    // shift = input layer average
    // width = sqrt(input layer gradient metric maximum)
    CL::utility::matrix<at::Tensor> x1_avg(NStates), s1_max(NStates),
                                    x2_avg(NStates), s2_max(NStates);
    // accumulate 0th example
    const auto & example = regset->examples()[0];
    const auto & x1s = example->x1s();
    const auto & Jx1rTs = example->Jx1rTs();
    const auto & x2s = example->x2s();
    const auto & Jx2rTs = example->Jx2rTs();
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = i; j < NStates; j++) {
        // input layer statistics
        const at::Tensor & x1 = x1s[i][j];
        x1_avg[i][j] = x1.clone();
        const at::Tensor & x2 = x2s[i][j];
        x2_avg[i][j] = x2.clone();
        // input layer gradient metric statistics
        const at::Tensor & J1T = Jx1rTs[i][j];
        s1_max[i][j] = J1T.transpose(0, 1).mm(J1T).diag();
        const at::Tensor & J2T = Jx2rTs[i][j];
        s2_max[i][j] = J2T.transpose(0, 1).mm(J2T).diag();
    }
    // accumulate remaining examples
    for (size_t iexample = 1; iexample < NExamples; iexample++) {
        const auto & example = regset->examples()[iexample];
        const auto & x1s = example->x1s();
        const auto & Jx1rTs = example->Jx1rTs();
        const auto & x2s = example->x2s();
        const auto & Jx2rTs = example->Jx2rTs();
        for (size_t i = 0; i < NStates; i++)
        for (size_t j = i; j < NStates; j++) {
            // input layer statistics
            const at::Tensor & x1 = x1s[i][j];
            x1_avg[i][j] += x1;
            const at::Tensor & x2 = x2s[i][j];
            x2_avg[i][j] += x2;
            // input layer gradient metric statistics
            const at::Tensor & J1T = Jx1rTs[i][j];
            at::Tensor S1 = J1T.transpose(0, 1).mm(J1T).diag();
            for (size_t k = 0; k < S1.size(0); k++) {
                s1_max[i][j][k].fill_(std::max(S1[k].item<double>(), s1_max[i][j][k].item<double>()));
            }
            const at::Tensor & J2T = Jx2rTs[i][j];
            at::Tensor S2 = J2T.transpose(0, 1).mm(J2T).diag();
            for (size_t k = 0; k < S2.size(0); k++) {
                s2_max[i][j][k].fill_(std::max(S2[k].item<double>(), s2_max[i][j][k].item<double>()));
            }
        }
    }
    // final process
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = i; j < NStates; j++) {
        // input layer average
        x1_avg[i][j] /= (double)NExamples;
        x2_avg[i][j] /= (double)NExamples;
        // For asymmetric irreducibles, the average is always 0 because
        // the network design has considered the symmetry:
        // * On one hand, this has effectively performed a data augmentation
        //   that considers all symmetry-equivalent geometries,
        //   and the asymmetric average of all those geometires is always 0
        //   This would not affect standard deviation, since (input layer)^2 is symmetric
        // * On the other hand, this excludes bias from asymmetric network,
        //   so it is impossible for the network to output a same value after x -= avg
        if (Hdnet1->irreds()[i][j] != 0) x1_avg[i][j].fill_(0.0);
        if (Hdnet2->irreds()[i][j] != 0) x2_avg[i][j].fill_(0.0);
        // sqrt(input layer gradient metric maximum)
        s1_max[i][j].sqrt_();
        s2_max[i][j].sqrt_();
    }
    return std::make_tuple(x1_avg, s1_max, x2_avg, s2_max);
}