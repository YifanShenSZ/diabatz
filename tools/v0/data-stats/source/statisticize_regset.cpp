#include "../include/global.hpp"
#include "../include/data.hpp"

std::tuple<
CL::utility::matrix<at::Tensor>, CL::utility::matrix<at::Tensor>,
CL::utility::matrix<at::Tensor>, CL::utility::matrix<at::Tensor>,
CL::utility::matrix<at::Tensor>, CL::utility::matrix<at::Tensor>,
CL::utility::matrix<at::Tensor>, CL::utility::matrix<at::Tensor>
>
statisticize_regset(const std::shared_ptr<abinitio::DataSet<RegHam>> & regset) {
    size_t NExamples = regset->size_int(),
           NStates = Hdnet->NStates();
    // we want: input layer average, minimum, maximum, standard deviation
    //          input layer gradient metric average, minimum, maximum, standard deviation
    CL::utility::matrix<at::Tensor> x_avg(NStates), x_min(NStates), x_max(NStates), x_std(NStates),
                                    S_avg(NStates), S_min(NStates), S_max(NStates), S_std(NStates);
    // accumulate 0th example
    const auto & example = regset->examples()[0];
    const auto & xs = example->xs();
    const auto & JxrTs = example->JxrTs();
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = i; j < NStates; j++) {
        // input layer statistics
        const at::Tensor & x = xs[i][j];
        x_avg[i][j] = x.clone();
        x_min[i][j] = x.clone();
        x_max[i][j] = x.clone();
        x_std[i][j] = x * x;
        // input layer gradient metric statistics
        const at::Tensor & JT = JxrTs[i][j];
        at::Tensor S = JT.transpose(0, 1).mm(JT).diag();
        S_avg[i][j] = S.clone();
        S_min[i][j] = S.clone();
        S_max[i][j] = S.clone();
        S_std[i][j] = S * S;
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
            for (size_t k = 0; k < x.numel(); k++) {
                x_min[i][j][k].fill_(std::min(x[k].item<double>(), x_min[i][j][k].item<double>()));
                x_max[i][j][k].fill_(std::max(x[k].item<double>(), x_max[i][j][k].item<double>()));
            }
            x_std[i][j] += x * x;
            // input layer gradient metric statistics
            const at::Tensor & JT = JxrTs[i][j];
            at::Tensor S = JT.transpose(0, 1).mm(JT).diag();
            S_avg[i][j] += S;
            for (size_t k = 0; k < S.size(0); k++) {
                S_min[i][j][k].fill_(std::min(S[k].item<double>(), S_min[i][j][k].item<double>()));
                S_max[i][j][k].fill_(std::max(S[k].item<double>(), S_max[i][j][k].item<double>()));
            }
            S_std[i][j] += S * S;
        }
    }
    // compute average
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
        // input layer standard deviation
        x_std[i][j] /= (double)NExamples;
        x_std[i][j] -= x_avg[i][j] * x_avg[i][j];
        x_std[i][j].sqrt_();
        // input layer gradient metric average
        S_avg[i][j] /= (double)NExamples;
        // input layer gradient metric standard deviation
        S_std[i][j] /= (double)NExamples;
        S_std[i][j] -= S_avg[i][j] * S_avg[i][j];
        S_std[i][j].sqrt_();
    }
    return std::make_tuple(x_avg, x_min, x_max, x_std,
                           S_avg, S_min, S_max, S_std);
}

void print_regset_statistics(const std::shared_ptr<abinitio::DataSet<RegHam>> & regset) {
    size_t NStates = Hdnet->NStates();
    CL::utility::matrix<at::Tensor> x_avg, x_min, x_max, x_std,
                                    S_avg, S_min, S_max, S_std;
    std::tie(x_avg, x_min, x_max, x_std, S_avg, S_min, S_max, S_std) = statisticize_regset(regset);
    std::ofstream ofs_x; ofs_x.open("regset_x.txt");
    std::ofstream ofs_S; ofs_S.open("regset_S.txt");
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = i; j < NStates; j++) {
        // standard output
        std::cout << "state " << i + 1 << " - state " << j + 1 << '\n';
        std::cout << "* x      average       range [" << x_avg[i][j].min().item<double>() << ", " << x_avg[i][j].max().item<double>() << "]\n";
        std::cout << "* x standard deviation range [" << x_std[i][j].min().item<double>() << ", " << x_std[i][j].max().item<double>() << "]\n";
        std::cout << "* S      average       range [" << S_avg[i][j].min().item<double>() << ", " << S_avg[i][j].max().item<double>() << "]\n";
        std::cout << "* S standard deviation range [" << S_std[i][j].min().item<double>() << ", " << S_std[i][j].max().item<double>() << "]\n";
        // input layer details
        ofs_x << "state " << i + 1 << " - state " << j + 1 << ", average, minimum, maximum, standard deviation\n";
        for (size_t k = 0; k < x_avg[i][j].numel(); k++)
        ofs_x << std::setw(10) << k + 1
              << std::setw(25) << std::scientific << std::setprecision(15) << x_avg[i][j][k].item<double>()
              << std::setw(25) << std::scientific << std::setprecision(15) << x_min[i][j][k].item<double>()
              << std::setw(25) << std::scientific << std::setprecision(15) << x_max[i][j][k].item<double>()
              << std::setw(25) << std::scientific << std::setprecision(15) << x_std[i][j][k].item<double>() << '\n';
        // input layer gradient metric details
        ofs_S << "state " << i + 1 << " - state " << j + 1 << ", average, minimum, maximum, standard deviation\n";
        for (size_t k = 0; k < S_avg[i][j].size(0); k++)
        ofs_S << std::setw(10) << k + 1
              << std::setw(25) << std::scientific << std::setprecision(15) << S_avg[i][j][k].item<double>()
              << std::setw(25) << std::scientific << std::setprecision(15) << S_min[i][j][k].item<double>()
              << std::setw(25) << std::scientific << std::setprecision(15) << S_max[i][j][k].item<double>()
              << std::setw(25) << std::scientific << std::setprecision(15) << S_std[i][j][k].item<double>() << '\n';
    }
    std::cout << "Details can be found in regset_x.txt and regset_S.txt\n";
    ofs_x.close(); ofs_S.close();
}