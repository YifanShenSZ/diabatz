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
    // Return a 0 if empty batch
    if (batch_size == 0) return at::zeros(1, c10::TensorOptions().dtype(torch::kFloat64));
    else {
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
            at::Tensor   Hd = Hdnets[thread]->forward(xs);
            at::Tensor DqHd = Hderiva::DxHd(Hd, xs, data->JxqTs());
            // Stop autograd tracking
            Hd.detach_();
            // Get adiabatic representation
            at::Tensor energy, states;
            std::tie(energy, states) = define_adiabatz(Hd, DqHd,
                data->JqrT(), data->cartdim(), data->NStates(), data->dH());
            // Make prediction in adiabatic representation
            int64_t NStates_data = data->NStates();
            energy = energy.slice(0, 0, NStates_data);
            at::Tensor DqHa = tchem::linalg::UT_sy_U(DqHd, states);
            CL::utility::matrix<at::Tensor> SADQHa(NStates_data);
            for (size_t i = 0; i < NStates_data; i++)
            for (size_t j = i; j < NStates_data; j++)
            SADQHa[i][j] = data->C2Qs(data->irreds(i, j)).mv(data->JqrT().mv(DqHa[i][j]));
            // energy residue
            std::vector<at::Tensor> r;
            r.push_back(unit * (energy - data->energy()));
            for (size_t i = 0; i < NStates_data; i++) r[0][i] *= data->sqrtweight_E(i);
            // (▽H)a residue
            for (size_t i = 0; i < NStates_data; i++)
            for (size_t j = i; j < NStates_data; j++)
            r.push_back(data->sqrtweight_dH(i, j) * data->sqrtSQs(data->irreds(i, j)).mv(SADQHa[i][j] - data->SAdH(i, j)));
            // total residue
            residues[idata] = at::cat(r);
        }
        return at::cat(residues);
    }
}

at::Tensor deg_residue(const std::vector<std::shared_ptr<DegHam>> & batch) {
    size_t batch_size = batch.size();
    // Return a 0 if empty batch
    if (batch_size == 0) return at::zeros(1, c10::TensorOptions().dtype(torch::kFloat64));
    else {
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
            at::Tensor   Hd = Hdnets[thread]->forward(xs);
            at::Tensor DqHd = Hderiva::DxHd(Hd, xs, data->JxqTs());
            // Stop autograd tracking
            Hd.detach_();
            // Get composite representation
            at::Tensor eigval, eigvec;
            std::tie(eigval, eigvec) = define_composite(Hd, DqHd,
                data->JqrT(), data->cartdim(), data->H(), data->dH());
            // Make prediction in composite representation
            at::Tensor   Hc = tchem::linalg::UT_sy_U(  Hd, eigvec);
            at::Tensor DqHc = tchem::linalg::UT_sy_U(DqHd, eigvec);
            CL::utility::matrix<at::Tensor> SADQHc(NStates);
            for (size_t i = 0; i < NStates; i++)
            for (size_t j = i; j < NStates; j++)
            SADQHc[i][j] = data->C2Qs(data->irreds(i, j)).mv(data->JqrT().mv(DqHc[i][j]));
            // Hc residue
            std::vector<at::Tensor> r;
            at::Tensor r_H = unit * (Hc - data->H());
            for (size_t i = 0; i < NStates; i++)
            for (size_t j = i; j < NStates; j++)
            if (data->irreds(i, j) == 0) {
                r.push_back(data->sqrtweight_H(i, j) * r_H[i][j]);
                r.back().resize_({1});
            }
            // (▽H)c residue
            for (size_t i = 0; i < NStates; i++)
            for (size_t j = i; j < NStates; j++)
            r.push_back(data->sqrtweight_dH(i, j) * data->sqrtSQs(data->irreds(i, j)).mv(SADQHc[i][j] - data->SAdH(i, j)));
            // total residue
            residues[idata] = at::cat(r);
        }
        return at::cat(residues);
    }
}

at::Tensor energy_residue(const std::vector<std::shared_ptr<Energy>> & batch) {
    size_t batch_size = batch.size();
    // Return a 0 if empty batch
    if (batch_size == 0) return at::zeros(1, c10::TensorOptions().dtype(torch::kFloat64));
    else {
        std::vector<at::Tensor> residues(batch_size);
        #pragma omp parallel for
        for (size_t idata = 0; idata < batch_size; idata++) {
            int thread = omp_get_thread_num();
            const auto & data = batch[idata];
            // get energy
            CL::utility::matrix<at::Tensor> xs = data->xs();
            at::Tensor Hd = Hdnets[thread]->forward(xs);
            Hd.detach_();
            at::Tensor energy, states;
            std::tie(energy, states) = Hd.symeig();
            // energy residue
            int64_t NStates_data = data->NStates();
            energy = energy.slice(0, 0, NStates_data);
            at::Tensor r = unit * (energy - data->energy());
            for (size_t i = 0; i < data->NStates(); i++) r[i] *= data->sqrtweight_E(i);
            // total residue
            residues[idata] = r;
        }
        return at::cat(residues);
    }
}

} // namespace torch_optim
} // namespace train