#include <tchem/linalg.hpp>
#include <tchem/chemistry.hpp>

#include "global.hpp"

void compare() {
    c10::TensorOptions top = at::TensorOptions().dtype(torch::kFloat64);
    int64_t NStates = Hdkernel->NStates();
    std::vector<std::shared_ptr<tchem::chem::Phaser>> phasers(NStates + 1);
    for (size_t i = 2; i < phasers.size(); i++)
    phasers[i] = std::make_shared<tchem::chem::Phaser>(i);
    // Data in adiabatic representation
    at::Tensor rmsd_energy = at::zeros(NStates, top),
               rmsd_dHa    = at::zeros({NStates, NStates}, top);
    std::vector<size_t> state_count(NStates, 0);
    for (auto & data : regset) {
        int64_t NStates = data->NStates();
        at::Tensor dH = data->dH();
        // Get necessary diabatic quantity
        at::Tensor Hd, dHd;
        std::tie(Hd, dHd) = Hdkernel->compute_Hd_dHd(data->geom());
        // Make prediction in adiabatic representation
        at::Tensor energy, states;
        std::tie(energy, states) = Hd.symeig(true);
        energy = energy.slice(0, 0, NStates);
        at::Tensor dHa = tchem::linalg::UT_sy_U(dHd, states);
        dHa = dHa.slice(0, 0, NStates).slice(1, 0, NStates);
        phasers[NStates]->fix_ob_(dHa, dH);
        // Count deviation
        for (size_t i = 0; i < NStates; i++) state_count[i] += 1;
        at::Tensor view_rmsd_energy = rmsd_energy.slice(0, 0, NStates);
        view_rmsd_energy += (energy - data->energy()).pow_(2);
        at::Tensor view_rmsd_dHa = rmsd_dHa.slice(0, 0, NStates).slice(1, 0, NStates);
        view_rmsd_dHa += (dHa - dH).pow_(2)
                         .transpose_(1, 2).transpose_(0, 1).sum_to_size({NStates, NStates});
    }
    for (size_t i = 0; i < NStates; i++) rmsd_energy[i] /= (double)state_count[i];
    rmsd_energy.sqrt_();
    rmsd_dHa /= (double)regset[0]->cartdim();
    for (size_t j = 0; j < NStates; j++)
    for (size_t i = 0; i <= j; i++)
    rmsd_dHa[i][j] /= (double)state_count[j];
    rmsd_dHa.sqrt_();
    std::cout << "Root mean square deviation of energy:\n";
    for (size_t i = 0; i < NStates; i++)
    std::cout << "State " << i + 1 << " = " << rmsd_energy[i].item<double>() << '\n';
    std::cout << "Root mean square deviation of energy gradient:\n";
    for (size_t i = 0; i < NStates; i++)
    std::cout << "State " << i + 1 << " = " << rmsd_dHa[i][i].item<double>() << '\n';
    std::cout << "Root mean square deviation of interstate coupling:\n";
    for (size_t i = 0    ; i < NStates; i++)
    for (size_t j = i + 1; j < NStates; j++)
    std::cout << "State " << i + 1 << "-" << j + 1 << " = " << rmsd_dHa[i][j].item<double>() << '\n';
}