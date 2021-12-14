#include <Foptim/line-search_2nd/BFGS.hpp>

#include "../include/global.hpp"
#include "../include/Hd_extension.hpp"

namespace {

at::Tensor init_guess_;

double lambda, miu;

void L(double & L, const double * free_intgeom, const int32_t & free_intdim) {
    // adiabatz
    at::Tensor q_free = at::from_blob(const_cast<double *>(free_intgeom), free_intdim,
                                      at::TensorOptions().dtype(torch::kFloat64));
    at::Tensor q = fixed_intcoord->vector_free2total(q_free);
    at::Tensor r = int2cart(q, init_guess_, intcoordset);
    at::Tensor energy = compute_energy(r);
    // L
    double T = (energy[target_state + 1] + energy[target_state]).item<double>(),
           C = (energy[target_state + 1] - energy[target_state]).item<double>();
    C = 0.5 * C * C;
    L = T - lambda * C + 0.5 * miu * C * C;
}

void L_Ld(double & L, double * Ld, const double * free_intgeom, const int32_t & free_intdim) {
    // adiabatz
    at::Tensor q_free = at::from_blob(const_cast<double *>(free_intgeom), free_intdim,
                                      at::TensorOptions().dtype(torch::kFloat64));
    at::Tensor q = fixed_intcoord->vector_free2total(q_free);
    at::Tensor r = int2cart(q, init_guess_, intcoordset);
    at::Tensor energy, dH;
    std::tie(energy, dH) = compute_energy_dHa(r);
    // L
    double T = (energy[target_state + 1] + energy[target_state]).item<double>(),
           C = (energy[target_state + 1] - energy[target_state]).item<double>();
    C = 0.5 * C * C;
    L = T - lambda * C + 0.5 * miu * C * C;
    // ▽T
    at::Tensor cartdT = dH[target_state + 1][target_state + 1] + dH[target_state][target_state];
    at::Tensor  intdT = intcoordset->gradient_cart2int(r, cartdT);
    at::Tensor free_intdT = fixed_intcoord->vector_total2free(intdT);
    // ▽C
    at::Tensor cartdC = (energy[target_state + 1] - energy[target_state])
                      * (dH[target_state + 1][target_state + 1] - dH[target_state][target_state]);
    at::Tensor  intdC = intcoordset->gradient_cart2int(r, cartdC);
    at::Tensor free_intdC = fixed_intcoord->vector_total2free(intdC);
    // ▽L
    at::Tensor free_intdL = free_intdT + (miu * C - lambda) * free_intdC;
    std::memcpy(Ld, free_intdL.data_ptr<double>(), free_intdim * sizeof(double));
}

void Ldd(double * Ldd, const double * free_intgeom, const int32_t & free_intdim) {
    // adiabatz
    at::Tensor q_free = at::from_blob(const_cast<double *>(free_intgeom), free_intdim,
                                      at::TensorOptions().dtype(torch::kFloat64));
    at::Tensor q = fixed_intcoord->vector_free2total(q_free);
    at::Tensor r = int2cart(q, init_guess_, intcoordset);
    at::Tensor energy, dH;
    std::tie(energy, dH) = compute_energy_dHa(r);
    at::Tensor ddH = compute_ddHa(r);
    // ▽▽T
    at::Tensor cartdT  = dH[target_state + 1][target_state + 1] + dH[target_state][target_state];
    at::Tensor cartddT = ddH[target_state + 1][target_state + 1] + ddH[target_state][target_state];
    at::Tensor  intddT = intcoordset->Hessian_cart2int(r, cartdT, cartddT);
    at::Tensor free_intddT = fixed_intcoord->matrix_total2free(intddT);
    // ▽▽C
    double C = (energy[target_state + 1] - energy[target_state]).item<double>();
    C = 0.5 * C * C;
    at::Tensor   Ediff = energy[target_state + 1] - energy[target_state],
                dEdiff =  dH[target_state + 1][target_state + 1] -  dH[target_state][target_state],
               ddEdiff = ddH[target_state + 1][target_state + 1] - ddH[target_state][target_state];
    at::Tensor cartdC = Ediff * dEdiff;
    at::Tensor  intdC = intcoordset->gradient_cart2int(r, cartdC);
    at::Tensor free_intdC = fixed_intcoord->vector_total2free(intdC);
    at::Tensor cartddC = dEdiff.outer(dEdiff) + Ediff * ddEdiff;
    at::Tensor  intddC = intcoordset->Hessian_cart2int(r, cartdC, cartddC);
    at::Tensor free_intddC = fixed_intcoord->matrix_total2free(intddC);
    // ▽▽L
    at::Tensor free_intddL = free_intddT + miu * free_intdC.outer(free_intdC) + (miu * C - lambda) * free_intddC;
    std::memcpy(Ldd, free_intddL.data_ptr<double>(), free_intdim * free_intdim * sizeof(double));
}

}

at::Tensor search_mex(const at::Tensor & _init_guess) {
    init_guess_ = _init_guess;
    at::Tensor q = (*intcoordset)(_init_guess);
    at::Tensor q_free = fixed_intcoord->vector_total2free(q);
    // Compute energy gap
    at::Tensor r = int2cart(q, init_guess_, intcoordset);
    at::Tensor energy = compute_energy(r);
    double avg = (energy[target_state + 1] + energy[target_state]).item<double>(),
           gap = (energy[target_state + 1] - energy[target_state]).item<double>();
    // Augmented Lagrangian
    double init_gap = std::max(gap, 1e-4);
    double init_C   = 0.5 * init_gap * init_gap;
    double init_miu = avg / pow(init_gap, 4.0);
    lambda = -init_miu * init_C;
    miu    =  init_miu;
    double switcher = sqrt(avg / miu);
    size_t iIteration = 0;
    std::cout << '\n';
    while (true) {
        // Minimize current augmented Lagrangian
        Foptim::BFGS(L, L_Ld, Ldd,
                     q_free.data_ptr<double>(), q_free.size(0),
                     20, 100, 1e-6, 1e-15);
        // Compute energy gap
        at::Tensor q = fixed_intcoord->vector_free2total(q_free);
        at::Tensor r = int2cart(q, init_guess_, intcoordset);
        at::Tensor energy = compute_energy(r);
        double gap = (energy[target_state + 1] - energy[target_state]).item<double>();
        // Check convergence
        if (gap < 1e-6) break;
        iIteration++;
        if (iIteration > 100) {
            std::cout << "Max iteration exceeds\n";
            break;
        }
        std::cout << "Iteration " << iIteration << ":\n"
                  << "lower state energy = " << energy[target_state].item<double>() << '\n'
                  << "gap = " << gap << '\n'
                  << "lamda = " << lambda << '\n'
                  << "miu = " << miu << '\n';
        std::cout << "Current internal coordinate =\n";
        for (size_t i = 0; i < q.size(0); i++)
        std::cout << std::scientific << std::setprecision(15) << q[i].item<double>() << '\n';
        std::cout << std::endl;
        // Get ready for next iteration
        double C = 0.5 * gap * gap;
        if (C < switcher) {
            std::cout << "Update lambda\n";
            lambda -= miu * C;
            switcher = switcher / pow(miu / init_miu, 0.9);
        }
        else {
            std::cout << "Update miu\n";
            miu *= 100.0;
            switcher /= 10.0;
        }
    }
    q = fixed_intcoord->vector_free2total(q_free);
    return int2cart(q, _init_guess, intcoordset);
}