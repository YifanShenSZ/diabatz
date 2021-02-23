#include <abinitio/Hamiltonian.hpp>

#include "global.hpp"

void map(const std::shared_ptr<abinitio::Geometry> & ham) {
    at::Tensor r = ham->geom();
    std::vector<at::Tensor> qs, Js;
    std::tie(qs, Js) = cart2int(r);
    size_t count = 0;
    std::vector<bool> qNotZero(qs.size());
    for (size_t i = 0; i < qs.size(); i++) {
        qNotZero[i] = qs[i].norm().item<double>() > threshold;
        if (qNotZero[i]) count++;
    }
    if (count == 0 || count == 1) {
        std::cout << "The point group is isomorphic to the CNPI group\n";
    }
    else if (count == qs.size()) {
        std::cout << "No symmetry, C1 point group\n";
    }
    else {
        std::cout << "The nonzero CNPI group irreducibles are:\n";
        for (size_t i = 0; i < qs.size(); i++)
        if (qNotZero[i]) std::cout << i << ' ';
        std::cout << '\n';
    }
}