#include <tchem/chemistry.hpp>

#include <Hd/kernel.hpp>

void int2normal(const Hd::kernel & Hdkernel, const tchem::chem::SANormalMode & final_vib) {
    auto para_mat = Hdkernel.Hdnet()->parameters();
    auto LTs = final_vib.intmodes();
    std::vector<at::Tensor> Ls(LTs.size());
    for (size_t i = 0; i < LTs.size(); i++) Ls[i] = LTs[i].transpose(0, 1).clone();
    // Loop over each independent matrix element
    for (size_t i = 0; i < Hdkernel.NStates(); i++)
    for (size_t j = i; j < Hdkernel.NStates(); j++) {
        const auto & sapset = (*Hdkernel.input_generator())[{i, j}];
        at::Tensor T = sapset.rotation(Ls);
        at::Tensor Tinv = T.inverse();
        at::Tensor c = Tinv.mv(para_mat[i][j][0].view(para_mat[i][j][0].numel()));
        std::string file = "Hd_" + std::to_string(i + 1) + "-" + std::to_string(j + 1) + ".txt";
        std::cout << "Hd" << i + 1 << j + 1 << " expanded in final-state normal modes can be found in " << file << '\n';
        std::ofstream ofs; ofs.open(file); {
            // constant (bias) term
            if (para_mat[i][j].size() > 1)
            ofs << 0 << '\n'
                << std::scientific << std::setw(25) << std::setprecision(15) << para_mat[i][j][1].item<double>() << '\n';
            // 1st and higher order terms
            for (size_t k = 0; k < c.size(0); k++) {
                sapset[k].pretty_print(ofs);
                ofs << std::scientific << std::setw(25) << std::setprecision(15) << c[k].item<double>() << '\n';
            }
        }
        ofs.close();
    }
}