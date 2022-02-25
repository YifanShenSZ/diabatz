#include "../include/global.hpp"
#include "../include/data.hpp"

void statisticize(
const std::shared_ptr<abinitio::DataSet<RegHam>> & regset,
const std::shared_ptr<abinitio::DataSet<DegHam>> & degset,
const std::shared_ptr<abinitio::DataSet<Energy>> & energy_set) {
    CL::utility::matrix<at::Tensor> avg_reg   , std_reg   ,
                                    avg_deg   , std_deg   ,
                                    avg_energy, std_energy;
    std::tie(avg_reg   , std_reg   ) = statisticize_input(regset    );
    std::tie(avg_deg   , std_deg   ) = statisticize_input(degset    );
    std::tie(avg_energy, std_energy) = statisticize_input(energy_set);
    // regular data statistics
    std::ofstream ofs; ofs.open("regset.txt");
    std::cout << "Regular data:\n";
    for (size_t i = 0; i < avg_reg.size(0); i++)
    for (size_t j = i; j < avg_reg.size(1); j++) {
        const at::Tensor & avg = avg_reg[i][j];
        const at::Tensor & std = std_reg[i][j];
        std::cout << "state " << i + 1 << " - state " << j + 1 << '\n';
        std::cout << "* average range [" << avg.min().item<double>() << ", " << avg.max().item<double>() << "]\n";
        std::cout << "* standard deviation range [" << std.min().item<double>() << ", " << std.max().item<double>() << "]\n";
        ofs << "state " << i + 1 << " - state " << j + 1 << " average and standard deviation\n";
        for (size_t k = 0; k < avg.numel(); k++)
        ofs << std::setw(10) << k + 1
            << std::setw(25) << std::scientific << std::setprecision(15) << avg[k].item<double>()
            << std::setw(25) << std::scientific << std::setprecision(15) << std[k].item<double>() << '\n';
    }
    std::cout << "Details can be found in regset.txt\n\n";
    ofs.close();
    // degenerate data statistics
    ofs.open("degset.txt");
    std::cout << "Degenerate data:\n";
    for (size_t i = 0; i < avg_deg.size(0); i++)
    for (size_t j = i; j < avg_deg.size(1); j++) {
        const at::Tensor & avg = avg_deg[i][j];
        const at::Tensor & std = std_deg[i][j];
        std::cout << "state " << i + 1 << " - state " << j + 1 << '\n';
        std::cout << "* average range [" << avg.min().item<double>() << ", " << avg.max().item<double>() << "]\n";
        std::cout << "* standard deviation range [" << std.min().item<double>() << ", " << std.max().item<double>() << "]\n";
        ofs << "state " << i + 1 << " - state " << j + 1 << " average and standard deviation\n";
        for (size_t k = 0; k < avg.numel(); k++)
        ofs << std::setw(10) << k + 1
            << std::setw(25) << std::scientific << std::setprecision(15) << avg[k].item<double>()
            << std::setw(25) << std::scientific << std::setprecision(15) << std[k].item<double>() << '\n';
    }
    std::cout << "Details can be found in degset.txt\n\n";
    ofs.close();
    // energy-only data statistics
    ofs.open("energy_set.txt");
    std::cout << "Energy-only data:\n";
    for (size_t i = 0; i < avg_deg.size(0); i++)
    for (size_t j = i; j < avg_deg.size(1); j++) {
        const at::Tensor & avg = avg_deg[i][j];
        const at::Tensor & std = std_deg[i][j];
        std::cout << "state " << i + 1 << " - state " << j + 1 << '\n';
        std::cout << "* average range [" << avg.min().item<double>() << ", " << avg.max().item<double>() << "]\n";
        std::cout << "* standard deviation range [" << std.min().item<double>() << ", " << std.max().item<double>() << "]\n";
        ofs << "state " << i + 1 << " - state " << j + 1 << " average and standard deviation\n";
        for (size_t k = 0; k < avg.numel(); k++)
        ofs << std::setw(10) << k + 1
            << std::setw(25) << std::scientific << std::setprecision(15) << avg[k].item<double>()
            << std::setw(25) << std::scientific << std::setprecision(15) << std[k].item<double>() << '\n';
    }
    std::cout << "Details can be found in energy_set.txt\n";
    ofs.close();
}