#include "../include/global.hpp"

// rescale Hdnet parameters according to feature scaling
// so that with scaled features Hdnet still outputs a same value
void rescale_Hdnet(const CL::utility::matrix<at::Tensor> & avg, const CL::utility::matrix<at::Tensor> & std) {
    torch::NoGradGuard no_grad;
    auto pmat = Hdnet->parameters();
    for (size_t istate = 0     ; istate < Hdnet->NStates(); istate++)
    for (size_t jstate = istate; jstate < Hdnet->NStates(); jstate++) {
        auto & ps = pmat[istate][jstate];
        ps[0] *= std[istate][jstate];
        if (Hdnet->irreds()[istate][jstate] == 0) ps[1] += ps[0].mv(avg[istate][jstate] / std[istate][jstate]);
    }
}

// undo Hdnet parameters scaling
// so that with unscaled features Hdnet still outputs a same value
void unscale_Hdnet(const CL::utility::matrix<at::Tensor> & avg, const CL::utility::matrix<at::Tensor> & std) {
    torch::NoGradGuard no_grad;
    auto pmat = Hdnet->parameters();
    for (size_t istate = 0     ; istate < Hdnet->NStates(); istate++)
    for (size_t jstate = istate; jstate < Hdnet->NStates(); jstate++) {
        auto & ps = pmat[istate][jstate];
        if (Hdnet->irreds()[istate][jstate] == 0) ps[1] -= ps[0].mv(avg[istate][jstate] / std[istate][jstate]);
        ps[0] /= std[istate][jstate];
    }
}
