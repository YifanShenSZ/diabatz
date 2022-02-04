#include <omp.h>

#include <tchem/linalg.hpp>

#include <Hderiva/diabatic.hpp>
#include <Hderiva/adiabatic.hpp>
#include <Hderiva/composite.hpp>

#include "../../include/data.hpp"

#include "common.hpp"

namespace train { namespace torch_optim {

at::Tensor reg_gradient(const std::vector<std::shared_ptr<RegHam>> & batch) {
    size_t batch_size = batch.size();
    // parallelly compute gradient of each data point
    std::vector<at::Tensor> gradients(batch_size);
    #pragma omp parallel for
    for (size_t idata = 0; idata < batch_size; idata++) {
        int thread = omp_get_thread_num();
        const auto & data = batch[idata];
        // get necessary diabatic quantities
        CL::utility::matrix<at::Tensor> xs = data->xs();
        for (size_t i = 0; i < NStates; i++)
        for (size_t j = i; j < NStates; j++)
        xs[i][j].set_requires_grad(true);
        at::Tensor     Hd = Hdnets[thread]->forward(xs);
        at::Tensor   DqHd = Hderiva::DxHd(Hd, xs, data->JxqTs(), true);
        at::Tensor   DcHd = Hderiva::DcHd(Hd, Hdnets[thread]->elements->parameters());
        at::Tensor DcDqHd = Hderiva::DcDxHd(DqHd, Hdnets[thread]->elements->parameters());
        // stop autograd tracking
          Hd.detach_();
        DqHd.detach_();
        // add the pretrained part
          Hd += data->pretrained_Hd  ();
        DqHd += data->pretrained_DqHd();
        // get adiabatic representation
        at::Tensor energy, states;
        std::tie(energy, states) = define_adiabatz(Hd, DqHd,
            data->JqrT(), data->cartdim(), data->NStates(), data->dH());
        // predict in adiabatic representation
        int64_t NStates_data = data->NStates();
        energy = energy.slice(0, 0, NStates_data);
        at::Tensor DqHa = tchem::linalg::UT_sy_U(DqHd, states);
        CL::utility::matrix<at::Tensor> SADQHa(NStates_data);
        for (size_t i = 0; i < NStates_data; i++)
        for (size_t j = i; j < NStates_data; j++)
        SADQHa[i][j] = data->C2Qs(data->irreds(i, j)).mv(data->JqrT().mv(DqHa[i][j]));
        // compute fitting parameter gradient in adiabatic prediction
        at::Tensor DcHa = tchem::linalg::UT_sy_U(DcHd, states);
        at::Tensor DcDqHa = Hderiva::DcDxHa(DqHa, DcHd, DcDqHd, energy, states);
        CL::utility::matrix<at::Tensor> DcSADQHa(NStates_data);
        for (size_t i = 0; i < NStates_data; i++)
        for (size_t j = i; j < NStates_data; j++)
        DcSADQHa[i][j] = data->C2Qs(data->irreds(i, j)).mm(data->JqrT().mm(DcDqHa[i][j]));
        // energy residue and Jacobian
        std::vector<at::Tensor> r, J;
        r.push_back(unit * (energy - data->energy()));
        for (size_t i = 0; i < NStates_data; i++) {
            r[0][i] *= data->sqrtweight_E(i);
            J.push_back(data->sqrtweight_E(i) * unit * DcHa[i][i]);
            J.back().resize_({1, J.back().numel()});
        }
        // (▽H)a residue and Jacobian
        for (size_t i = 0; i < NStates_data; i++)
        for (size_t j = i; j < NStates_data; j++) {
            r.push_back(data->sqrtweight_dH(i, j) * data->sqrtSQs(data->irreds(i, j)).mv(SADQHa[i][j] - data->SAdH(i, j)));
            J.push_back(data->sqrtweight_dH(i, j) * data->sqrtSQs(data->irreds(i, j)).mm(DcSADQHa[i][j]));
        }
        // total gradient
        gradients[idata] = at::matmul(at::cat(r), at::cat(J));
    }
    // accumulate gradients
    at::Tensor gradient = std::accumulate(gradients.begin() + 1, gradients.end(), gradients[0]) / (double)batch_size;
    return gradient;
}

at::Tensor deg_gradient(const std::vector<std::shared_ptr<DegHam>> & batch) {
    size_t batch_size = batch.size();
    // parallelly compute gradient of each data point
    std::vector<at::Tensor> gradients(batch_size);
    #pragma omp parallel for
    for (size_t idata = 0; idata < batch_size; idata++) {
        int thread = omp_get_thread_num();
        const auto & data = batch[idata];
        // get necessary diabatic quantities
        CL::utility::matrix<at::Tensor> xs = data->xs();
        for (size_t i = 0; i < NStates; i++)
        for (size_t j = i; j < NStates; j++)
        xs[i][j].set_requires_grad(true);
        at::Tensor     Hd = Hdnets[thread]->forward(xs);
        at::Tensor   DqHd = Hderiva::DxHd(Hd, xs, data->JxqTs(), true);
        at::Tensor   DcHd = Hderiva::DcHd(Hd, Hdnets[thread]->elements->parameters());
        at::Tensor DcDqHd = Hderiva::DcDxHd(DqHd, Hdnets[thread]->elements->parameters());
        // stop autograd tracking
          Hd.detach_();
        DqHd.detach_();
        // add the pretrained part
          Hd += data->pretrained_Hd  ();
        DqHd += data->pretrained_DqHd();
        // get composite representation
        at::Tensor eigval, eigvec;
        std::tie(eigval, eigvec) = define_composite(Hd, DqHd,
            data->JqrT(), data->cartdim(), data->H(), data->dH());
        // predict in composite representation
        at::Tensor   Hc = tchem::linalg::UT_sy_U(  Hd, eigvec);
        at::Tensor DqHc = tchem::linalg::UT_sy_U(DqHd, eigvec);
        CL::utility::matrix<at::Tensor> SADQHc(NStates);
        for (size_t i = 0; i < NStates; i++)
        for (size_t j = i; j < NStates; j++)
        SADQHc[i][j] = data->C2Qs(data->irreds(i, j)).mv(data->JqrT().mv(DqHc[i][j]));
        // compute fitting parameter gradient in composite prediction
        at::Tensor DcHc, DcDqHc;
        std::tie(DcHc, DcDqHc) = Hderiva::DcHc_DcDxHc(Hc, DqHc,
            DqHd, DcHd, DcDqHd, eigval, eigvec, data->Sq());
        CL::utility::matrix<at::Tensor> DcSADQHc(NStates);
        for (size_t i = 0; i < NStates; i++)
        for (size_t j = i; j < NStates; j++)
        DcSADQHc[i][j] = data->C2Qs(data->irreds(i, j)).mm(data->JqrT().mm(DcDqHc[i][j]));
        // Hc residue and Jacobian
        std::vector<at::Tensor> r, J;
        at::Tensor r_H = unit * (Hc - data->H()),
                   J_H = unit * DcHc;
        for (size_t i = 0; i < NStates; i++)
        for (size_t j = i; j < NStates; j++)
        if (data->irreds(i, j) == 0) {
            r.push_back(data->sqrtweight_H(i, j) * r_H[i][j]);
            J.push_back(data->sqrtweight_H(i, j) * J_H[i][j]);
            r.back().resize_({1});
            J.back().resize_({1, J.back().numel()});
        }
        // (▽H)c residue and Jacobian
        for (size_t i = 0; i < NStates; i++)
        for (size_t j = i; j < NStates; j++) {
            r.push_back(data->sqrtweight_dH(i, j) * data->sqrtSQs(data->irreds(i, j)).mv(SADQHc[i][j] - data->SAdH(i, j)));
            J.push_back(data->sqrtweight_dH(i, j) * data->sqrtSQs(data->irreds(i, j)).mm(DcSADQHc[i][j]));
        }
        // total gradient
        gradients[idata] = at::matmul(at::cat(r), at::cat(J));
    }
    // accumulate gradients
    at::Tensor gradient = std::accumulate(gradients.begin() + 1, gradients.end(), gradients[0]) / (double)batch_size;
    return gradient;
}

at::Tensor energy_gradient(const std::vector<std::shared_ptr<Energy>> & batch) {
    size_t batch_size = batch.size();
    // parallelly compute gradient of each data point
    std::vector<at::Tensor> gradients(batch_size);
    #pragma omp parallel for
    for (size_t idata = 0; idata < batch_size; idata++) {
        int thread = omp_get_thread_num();
        const auto & data = batch[idata];
        // get energy and its gradient over fitting parameters
        CL::utility::matrix<at::Tensor> xs = data->xs();
        at::Tensor   Hd = Hdnets[thread]->forward(xs);
        at::Tensor DcHd = Hderiva::DcHd(Hd, Hdnets[thread]->elements->parameters());
        Hd.detach_();
        Hd += data->pretrained_Hd(); // add the pretrained part
        at::Tensor energy, states;
        std::tie(energy, states) = Hd.symeig(true);
        at::Tensor DcHa = tchem::linalg::UT_sy_U(DcHd, states);
        // energy residue and Jacobian
        int64_t NStates_data = data->NStates();
        energy = energy.slice(0, 0, NStates_data);
        at::Tensor r = unit * (energy - data->energy());
        std::vector<at::Tensor> J;
        for (size_t i = 0; i < NStates_data; i++) {
            r[i] *= data->sqrtweight_E(i);
            J.push_back(data->sqrtweight_E(i) * unit * DcHa[i][i]);
            J.back().resize_({1, J.back().numel()});
        }
        // total gradient
        gradients[idata] = at::matmul(r, at::cat(J));
    }
    // accumulate gradients
    at::Tensor gradient = std::accumulate(gradients.begin() + 1, gradients.end(), gradients[0]) / (double)batch_size;
    return gradient;
}

} // namespace torch_optim
} // namespace train