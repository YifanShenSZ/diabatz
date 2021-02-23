#include <tchem/linalg.hpp>

#include <Hderiva/Hderiva.hpp>

#include "global.hpp"
#include "data.hpp"
#include "train.hpp"

namespace train {

void Jacobian(double * JT, const double * c, const int32_t & M, const int32_t & N) {
    // Make a tensor J sharing memory with JT, since it is easier to manipulate J
    at::Tensor J = at::from_blob(JT, {N, M}, at::TensorOptions().dtype(torch::kFloat64));
    J.transpose_(0, 1);
    #pragma omp parallel for
    for (size_t thread = 0; thread < OMP_NUM_THREADS; thread++) {
        c2p(c, thread);
        size_t start = segstart[thread];
        for (const auto & data : regchunk[thread]) {
            at::Tensor JqrT = data->JqrT();
            std::vector<at::Tensor> sqrtSs = data->sqrtSs();
            double weight = data->weight();
            int64_t NStates = data->NStates();
            CL::utility::matrix<size_t> irreds = data->irreds();

            // Get necessary diabatic quantities
            CL::utility::matrix<at::Tensor> xs = data->xs();
            for (size_t i = 0; i < NStates; i++)
            for (size_t j = i; j < NStates; j++)
            xs[i][j].set_requires_grad(true);
            at::Tensor     Hd = Hdnets[thread]->forward(xs);
            at::Tensor   DqHd = Hderiva::DxHd(Hd, xs, data->JxqTs(), true);
            at::Tensor   DcHd = Hderiva::DcHd(Hd, Hdnets[thread]->elements->parameters());
            at::Tensor DcDqHd = Hderiva::DcDxHd(DqHd,  Hdnets[thread]->elements->parameters());
            // Stop autograd tracking
              Hd.detach_();
            DqHd.detach_();

            // Get adiabatic representation
            at::Tensor energy, states;
            std::tie(energy, states) = Hd.symeig(true);
            at::Tensor DqHa = tchem::linalg::UT_sy_U(DqHd, states);
            DqHa = DqHa.slice(0, 0, NStates).slice(1, 0, NStates);
            at::Tensor cartDqHa = DqHa.new_empty({NStates, NStates});
            for (size_t i = 0; i < NStates; i++)
            for (size_t j = i; j < NStates; j++)
            cartDqHa[i][j] = JqrT.mv(DqHa[i][j]);
            size_t iphase = phasers[NStates]->iphase_min(cartDqHa, data->dH());
            at::Tensor states_view = states.slice(1, 0, NStates);
            phasers[NStates]->alter_states_(states_view, iphase);

            // Compute fitting parameter gradient of adiabatic prediction
            at::Tensor DcHa = tchem::linalg::UT_sy_U(DcHd, states);
            DqHa = tchem::linalg::UT_sy_U(DqHd, states);
            at::Tensor DcDqHa = Hderiva::DcDxHa(DqHa, DcHd, DcDqHd, energy, states);
            CL::utility::matrix<at::Tensor> DcSADqHa(NStates);
            for (size_t i = 0; i < NStates; i++)
            for (size_t j = i; j < NStates; j++)
            DcSADqHa[i][j] = data->cat(data->split2CNPI(DcDqHa[i][j]))[irreds[i][j]];

            // energy Jacobian
            at::Tensor J_E = weight * unit * DcHa;
            for (size_t i = 0; i < NStates; i++) {
                J[start].copy_(J_E[i][i]);
                start++;
            }
            // (▽H)a Jacobian
            for (size_t i = 0; i < NStates; i++)
            for (size_t j = i; j < NStates; j++) {
                at::Tensor r_g = weight * sqrtSs[irreds[i][j]].mv(DcSADqHa[i][j]);
                size_t stop = start + r_g.size(0);
                J.slice(0, start, stop).copy_(r_g);
                start = stop;
            }
        }
    }
}

}