#include <tchem/chemistry.hpp>

#include <abinitio/SAHamiltonian.hpp>

namespace abinitio {

void RegSAHam::construct_dH_(const at::Tensor & cartdH) {
    std::vector<at::Tensor> Js_point = cat(Js_);
    std::vector<at::Tensor> cart2int(NPointIrreds());
    for (size_t i = 0; i < NPointIrreds(); i++) {
        at::Tensor JJT = Ss_[i];
        at::Tensor cholesky = JJT.cholesky(true);
        at::Tensor inverse = at::cholesky_inverse(cholesky, true);
        cart2int[i] = inverse.mm(Js_point[i]);
    }
    irreds_.resize(cartdH.size(0));
    dH_.resize(cartdH.size(0));
    for (size_t i = 0; i < dH_.size(0); i++) {
        // Diagonals must be totally symmetric
        irreds_[i][i] = 0;
        dH_[i][i] = cart2int[0].mm(cartdH[i][i]);
        // Try out every irreducible for off-diagonals
        for (size_t j = i + 1; j < dH_.size(1); j++) {
            size_t irred;
            for (irred = 0; irred < NPointIrreds(); irred++) {
                dH_[i][j] = cart2int[irred].mm(cartdH[i][j]);
                if (dH_[i][j].norm().item<double>() > 1e-6) {
                    irreds_[i][j] = irred;
                    break;
                }
            }
            if (irred == NPointIrreds()) throw "Vanishing gradient, cannot tell ▽H irreducible";
        }
    }
}

RegSAHam::RegSAHam() {}
// See the base class constructor for details of `cart2int`
RegSAHam::RegSAHam(const SAHamLoader & loader,
std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> (*cart2int)(const at::Tensor &))
: SAGeometry(loader.geom, loader.CNPI2point, cart2int) {
    energy_ = loader.energy.clone();
    at::Tensor cartdH = loader.dH.clone();
    for (size_t i = 0    ; i < cartdH.size(0); i++)
    for (size_t j = i + 1; j < cartdH.size(1); j++)
    cartdH[i][j] *= energy_[j] - energy_[i];
    this->construct_dH_(cartdH);
}
RegSAHam::~RegSAHam() {}

double RegSAHam::weight() const {return weight_;}
at::Tensor RegSAHam::energy() const {return energy_;}
CL::utility::matrix<size_t> RegSAHam::irreds() const {return irreds_;}
CL::utility::matrix<at::Tensor> RegSAHam::dH() const {return dH_;}

void RegSAHam::to(const c10::DeviceType & device) {
    SAGeometry::to(device);
    energy_.to(device);
    for (size_t i = 0; i < dH_.size(0); i++)
    for (size_t j = i; j < dH_.size(0); j++)
    dH_[i][j].to(device);
}

// Subtract zero point from energy
void RegSAHam::subtract_ZeroPoint(const double & zero_point) {
    energy_ -= zero_point;
}
// Lower the weight if energy[0] > thresh
void RegSAHam::adjust_weight(const double & thresh) {
    double temp = energy_[0].item<double>();
    if (temp > thresh) {
        temp = thresh / temp;
        weight_ = temp * temp;
    }
}





DegSAHam::DegSAHam() {}
// See the base class constructor for details of `cart2int`
DegSAHam::DegSAHam(const SAHamLoader & loader,
std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> (*cart2int)(const at::Tensor &))
: RegSAHam(loader, cart2int) {
    H_ = energy_.clone();
    at::Tensor cartdH = loader.dH.clone();
    for (size_t i = 0    ; i < cartdH.size(0); i++)
    for (size_t j = i + 1; j < cartdH.size(1); j++)
    cartdH[i][j] *= energy_[j] - energy_[i];
    tchem::chem::composite_representation(H_, cartdH);
    this->construct_dH_(cartdH);
    // point group symmetry consistency check
    for (size_t i = 0    ; i < cartdH.size(0); i++)
    for (size_t j = i + 1; j < cartdH.size(1); j++)
    if (H_[i][j].norm().item<double>() > 1e-6 && irreds_[i][j] != 0)
    throw "Inconsistent irreducible between H and ▽H";
}
DegSAHam::~DegSAHam() {}

at::Tensor DegSAHam::H() const {return H_;};

void DegSAHam::to(const c10::DeviceType & device) {
    RegSAHam::to(device);
    H_.to(device);
}

// Subtract zero point from energy and H
void DegSAHam::subtract_ZeroPoint(const double & zero_point) {
    RegSAHam::subtract_ZeroPoint(zero_point);
    H_ -= zero_point * at::eye(H_.size(0), H_.options());
}

} // namespace abinitio