#include <tchem/chemistry.hpp>

#include <Hd/kernel.hpp>

void int2normal(const Hd::kernel & Hdkernel, const tchem::chem::SANormalMode & final_vib, const at::Tensor & shift) {
    auto para_mat = Hdkernel.Hdnet()->parameters();
    auto LTs = final_vib.intmodes();
    std::vector<at::Tensor> Ls(LTs.size());
    for (size_t i = 0; i < LTs.size(); i++) Ls[i] = LTs[i].transpose(0, 1).clone();
    // Loop over each independent matrix element
    for (size_t i = 0; i < Hdkernel.NStates(); i++)
    for (size_t j = i; j < Hdkernel.NStates(); j++) {
        const auto & original_sapset = (*Hdkernel.input_generator())[{i, j}];
        auto sapset = * original_sapset;
        // Add the bias (const term) to sap set for rotation and translation
        if (para_mat[i][j].size() > 1) sapset.insert_const();
        at::Tensor Trota = sapset.rotation(Ls),
                   Ttran = sapset.translation(shift);
        at::Tensor c;
        // Concatenate the bias (const term) coefficient
        if (para_mat[i][j].size() > 1) c = at::cat({para_mat[i][j][1], para_mat[i][j][0].view(para_mat[i][j][0].numel())});
        else c = para_mat[i][j][0].view(para_mat[i][j][0].numel());
        c = Trota.transpose(0, 1).mv(Ttran.transpose(0, 1).mv(c));
        std::string file = "Hd_" + std::to_string(i + 1) + "-" + std::to_string(j + 1) + ".txt";
        std::cout << "Hd" << i + 1 << j + 1 << " expanded in final-state normal modes can be found in " << file << '\n';
        std::ofstream ofs; ofs.open(file); {
            // 1st and higher order terms
            for (size_t k = 0; k < c.size(0); k++) {
                const auto & sap = sapset[k];
                sap.pretty_print(ofs);
                bool diag = false;
                if (sap.order() == 2) if (sap[0] == sap[1]) diag = true;
                // Subtract harmonic term from diagonal
                if (diag) {
                    double sub = final_vib.frequencies()[sap[0].first][sap[0].second].item<double>();
                    sub = 0.5 * sub * sub;
                    ofs << std::scientific << std::setw(25) << std::setprecision(15)
                        << c[k].item<double>() - sub << '\n';
                }
                else ofs << std::scientific << std::setw(25) << std::setprecision(15) << c[k].item<double>() << '\n';
            }
        }
        ofs.close();
    }
}