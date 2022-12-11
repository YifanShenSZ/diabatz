#include <tchem/linalg.hpp>
#include <tchem/chemistry.hpp>

#include "../include/global.hpp"
#include "../include/data.hpp"

void print_degset_statistics(const std::shared_ptr<abinitio::DataSet<DegHam>> & degset) {
    size_t NStates = Hdnet->NStates();
    std::ofstream ofs; ofs.open("degset.txt");
    for (const auto& data : degset->examples()) {
        const at::Tensor& dH = data->dH();
        at::Tensor dHdH = tchem::linalg::sy3matdotmul(dH, dH);
        at::Tensor eigval, eigvec;
        std::tie(eigval, eigvec) = dHdH.symeig();
        // standard output
        if (tchem::chem::check_degeneracy(eigval, 0.0001)) {
            std::cout << data->path() << " has degenerate composite representation: ";
            for (int64_t i = 0; i < eigval.size(0); i++) std::cout << std::setw(13) << std::scientific << std::setprecision(3) << eigval[i].item<double>();
            std::cout << '\n';
        }
        // composite eigenvalue details
        ofs << data->path() << ": ";
        for (int64_t i = 0; i < eigval.size(0); i++) ofs << std::setw(13) << std::scientific << std::setprecision(3) << eigval[i].item<double>();
        ofs << '\n';
    }
    std::cout << "Details can be found in degset.txt\n";
    ofs.close();
}