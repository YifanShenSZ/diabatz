#include <tchem/linalg.hpp>

#include <Hderiva/Hderiva.hpp>

#include "global.hpp"
#include "data.hpp"
#include "train.hpp"

namespace train {

void residue(double * r, const double * c, const int32_t & M, const int32_t & N) {

#pragma omp parallel for
for (size_t thread = 0; thread < OMP_NUM_THREADS; thread++) {
    c2p(c, thread);
    size_t start = segstart[thread];
    for (size_t idata = regchunk[thread] - regchunk[0]; idata < regchunk[thread]; idata++) {
        auto & data = regset[idata];
        at::Tensor JqrT = data->JqrT();
        std::vector<at::Tensor> sqrtSs = data->sqrtSs();
        double weight = data->weight();
        int64_t NStates = data->NStates();
        CL::utility::matrix<size_t> irreds = data->irreds();
        CL::utility::matrix<at::Tensor> SAdH = data->SAdH();

        // Get necessary diabatic quantities
        CL::utility::matrix<at::Tensor> xs = data->xs();
        for (size_t i = 0; i < NStates; i++)
        for (size_t j = i; j < NStates; j++)
        xs[i][j].set_requires_grad(true);
        at::Tensor   Hd = Hdnets[thread]->forward(xs);
        at::Tensor DqHd = Hderiva::DxHd(Hd, xs, data->JxqTs(), false);
        // Stop autograd tracking
        Hd.detach_();

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
        phasers[NStates]->alter_ob_(DqHa, iphase);

        // Make prediction in adiabatic representation
        energy = energy.slice(0, 0, NStates);
        CL::utility::matrix<at::Tensor> SADqHa(NStates);
        for (size_t i = 0; i < NStates; i++)
        for (size_t j = i; j < NStates; j++)
        SADqHa[i][j] = data->cat(data->split2CNPI(DqHa[i][j]))[irreds[i][j]];

        // energy residue
        at::Tensor r_E = weight * unit * (energy - data->energy());
        std::memcpy(&(r[start]), r_E.data_ptr<double>(), NStates * sizeof(double));
        start += NStates;
        // ▽H residue
        for (size_t i = 0; i < NStates; i++)
        for (size_t j = i; j < NStates; j++) {
            at::Tensor r_g = weight * sqrtSs[irreds[i][j]].mv(SADqHa[i][j] - SAdH[i][j]);
            std::memcpy(&(r[start]), r_g.data_ptr<double>(), r_g.numel() * sizeof(double));
            start += r_g.numel();
        }
    }
}

}

}