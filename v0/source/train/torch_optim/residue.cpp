#include <omp.h>

#include <tchem/linalg.hpp>

#include <Hderiva/diabatic.hpp>
#include <Hderiva/adiabatic.hpp>
#include <Hderiva/composite.hpp>

#include "../../../include/data.hpp"

#include "../common.hpp"

namespace train { namespace torch_optim {

at::Tensor reg_residue(const std::vector<std::shared_ptr<RegHam>> & batch) {
    size_t batch_size = batch.size();
    std::vector<at::Tensor> residues(batch_size);
    #pragma omp parallel for
    for (size_t idata = 0; idata < batch_size; idata++) {
        int thread = omp_get_thread_num();
        const auto & data = batch[idata];
        // Get necessary diabatic quantities
        CL::utility::matrix<at::Tensor> xs = data->xs();
        for (size_t i = 0; i < NStates; i++)
        for (size_t j = i; j < NStates; j++)
        xs[i][j].set_requires_grad(true);
        at::Tensor     Hd = Hdnets[thread]->forward(xs);
        at::Tensor   DqHd = Hderiva::DxHd(Hd, xs, data->JxqTs(), true);
        // Stop autograd tracking
        Hd.detach_();
        // Get adiabatic representation
        at::Tensor energy, states;
        std::tie(energy, states) = define_adiabatz(Hd, DqHd,
            data->JqrT(), data->cartdim(), data->NStates(), data->dH());
        // Make prediction in adiabatic representation
        int64_t NStates_data = data->NStates();
        CL::utility::matrix<size_t> irreds = data->irreds();
        energy = energy.slice(0, 0, NStates_data);
        at::Tensor DqHa = tchem::linalg::UT_sy_U(DqHd, states);
        CL::utility::matrix<at::Tensor> SADqHa(NStates_data);
        for (size_t i = 0; i < NStates_data; i++)
        for (size_t j = i; j < NStates_data; j++)
        SADqHa[i][j] = data->cat(data->split2CNPI(DqHa[i][j]))[irreds[i][j]];
        // energy residue
        std::vector<at::Tensor> r;
        r.push_back(unit * (energy - data->energy()));
        for (size_t i = 0; i < NStates_data; i++) r[0][i] *= data->sqrtweight_E(i);
        // (▽H)a residue
        std::vector<at::Tensor> sqrtSs = data->sqrtSs();
        CL::utility::matrix<at::Tensor> SAdH = data->SAdH();
        for (size_t i = 0; i < NStates_data; i++)
        for (size_t j = i; j < NStates_data; j++)
        r.push_back(data->sqrtweight_dH(i, j) * sqrtSs[irreds[i][j]].mv(SADqHa[i][j] - SAdH[i][j]));
        // total residue
        residues[idata] = at::cat(r);
    }
    return at::cat(residues);
}

} // namespace torch_optim
} // namespace train