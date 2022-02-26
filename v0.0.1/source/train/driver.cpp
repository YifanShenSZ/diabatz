#include <Foptim/Foptim.hpp>

#include <CppLibrary/linalg.hpp>

#include "common.hpp"

namespace train { namespace line_search {

std::tuple<int32_t, int32_t> count_eq_par() {
    int32_t NEqs = 0;
    for (const auto & data : regset) {
        size_t NStates_data = data->NStates();
        // energy least square equations
        NEqs += NStates_data;
        // (▽H)a least square equations
        for (size_t i = 0; i < NStates_data; i++)
        for (size_t j = i; j < NStates_data; j++)
        NEqs += data->SAdH(i, j).size(0);
    }
    for (const auto & data : degset) {
        if (NStates != data->NStates()) throw std::invalid_argument(
        "Degenerate data must share a same number of electronic states with "
        "the model to define a comparable composite representation");
        // Hc least square equations
        for (size_t i = 0; i < NStates; i++)
        for (size_t j = i; j < NStates; j++)
        if (data->irreds(i, j) == 0) NEqs++;
        // (▽H)c least square equations
        for (size_t i = 0; i < NStates; i++)
        for (size_t j = i; j < NStates; j++)
        NEqs += data->SAdH(i, j).size(0);
    }
    for (const auto & data : energy_set) {
        // energy least square equations
        NEqs += data->NStates();
    }
    std::cout << "The data set corresponds to " << NEqs << " least square equations\n";

    int32_t NPars = 0;
    for (const auto & p : Hdnet->elements->parameters()) NPars += p.numel();
    std::cout << "There are " << NPars << " parameters to train\n\n";

    return std::make_tuple(NEqs, NPars);
}

void loss(double & l, const double * c, const int32_t & N);
void loss_gradient(double & l, double * g, const double * c, const int32_t & N);

void regularized_loss(double & l, const double * c, const int32_t & N);
void regularized_loss_gradient(double & l, double * g, const double * c, const int32_t & N);

void optimize(const bool & regularized, const std::string & optimizer, const double & learning_rate, const int32_t & memory, const size_t & max_iteration) {
    int32_t NEqs, NPars;
    std::tie(NEqs, NPars) = count_eq_par();

    double * c = new double[NPars];
    p2c(0, c);
    for (size_t thread = 0; thread < OMP_NUM_THREADS; thread++) gradients[thread] = at::zeros(NPars, at::TensorOptions().dtype(torch::kFloat64));
    // display initial loss
    double l;
    loss(l, c, NPars);
    std::cout << "The initial loss = " << l << std::endl;

    if (regularized) {
        if (optimizer == "GD")
        Foptim::steepest_descent_verbose(regularized_loss, regularized_loss_gradient,
            c, NPars,
            learning_rate, max_iteration);
        else if (optimizer == "CGDY")
        Foptim::CGDY_verbose(regularized_loss, regularized_loss_gradient,
            c, NPars,
            learning_rate, max_iteration);
        else if (optimizer == "CGPR")
        Foptim::CGPR_verbose(regularized_loss, regularized_loss_gradient,
            c, NPars,
            learning_rate, max_iteration);
        else if (optimizer == "LBFGS")
        Foptim::LBFGS_verbose(regularized_loss, regularized_loss_gradient,
            c, NPars,
            learning_rate, memory, max_iteration);
        else throw std::invalid_argument("Unsupported optimizer " + optimizer);
    }
    else {
        if (optimizer == "GD")
        Foptim::steepest_descent_verbose(loss, loss_gradient,
            c, NPars,
            learning_rate, max_iteration);
        else if (optimizer == "CGDY")
        Foptim::CGDY_verbose(loss, loss_gradient,
            c, NPars,
            learning_rate, max_iteration);
        else if (optimizer == "CGPR")
        Foptim::CGPR_verbose(loss, loss_gradient,
            c, NPars,
            learning_rate, max_iteration);
        else if (optimizer == "LBFGS")
        Foptim::LBFGS_verbose(loss, loss_gradient,
            c, NPars,
            learning_rate, memory, max_iteration);
        else throw std::invalid_argument("Unsupported optimizer " + optimizer);
    }
    c2p(c, 0);

    loss(l, c, NPars);
    std::cout << "The final loss = " << l << '\n';
    delete [] c;
}

} // namespace line_search
} // namespace train