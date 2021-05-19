#include <tchem/chemistry.hpp>

#include <abinitio/SAHamiltonian.hpp>

namespace abinitio {

// Construct `SAdH_` based on constructed `dH_`
// Determine `irreds_` by finding the largest segment of each `SAdH_` element
void RegSAHam::construct_symmetry_() {
    assert(("`dH_` must have been constructed", dH_.defined()));
    std::vector<at::Tensor> Js_point = this->cat(Jqrs_);
    std::vector<at::Tensor> cart2int(NPointIrreds());
    for (size_t i = 0; i < NPointIrreds(); i++) {
        at::Tensor JJT = Ss_[i];
        at::Tensor cholesky = JJT.cholesky(true);
        at::Tensor inverse = at::cholesky_inverse(cholesky, true);
        cart2int[i] = inverse.mm(Js_point[i]);
    }
    SAdH_.resize(dH_.size(0));
    irreds_.resize(dH_.size(0));
    for (size_t i = 0; i < dH_.size(0); i++) {
        // Diagonals must be totally symmetric
        SAdH_[i][i] = cart2int[0].mv(dH_[i][i]);
        irreds_[i][i] = 0;
        // Try out every irreducible for off-diagonals
        for (size_t j = i + 1; j < dH_.size(1); j++) {
            SAdH_[i][j] = cart2int[0].mv(dH_[i][j]);
            double infnorm = at::max(at::abs(SAdH_[i][j])).item<double>();
            irreds_[i][j] = 0;
            for (size_t irred = 1; irred < NPointIrreds(); irred++) {
                at::Tensor candidate = cart2int[irred].mv(dH_[i][j]);
                double cannorm = at::max(at::abs(candidate)).item<double>();
                if (cannorm > infnorm) {
                    SAdH_[i][j] = candidate;
                    infnorm = cannorm;
                    irreds_[i][j] = irred;
                }
            }
        }
    }
}
// Reconstruct `dH_` based on constructed `SAdH_`
// to eliminate the symmetry breaking flaw in original data
void RegSAHam::reconstruct_dH_() {
    assert(("`SAdH_` must have been constructed", ! SAdH_.empty()));
    std::vector<at::Tensor> JTs_point = cat(JqrTs_, 1);
    for (size_t i = 0; i < dH_.size(0); i++)
    for (size_t j = i; j < dH_.size(1); j++)
    dH_[i][j] = JTs_point[irreds_[i][j]].mv(SAdH_[i][j]);
}

RegSAHam::RegSAHam() {}
RegSAHam::RegSAHam(const RegSAHam & source) : SAGeometry(source),
energy_(source.energy_), dH_(source.dH_),
weight_E_(source.weight_E_), sqrtweight_E_(source.sqrtweight_E_),
weight_dH_(source.weight_dH_), sqrtweight_dH_(source.sqrtweight_dH_),
irreds_(source.irreds_), SAdH_(source.SAdH_) {}
// See the base class constructor for details of `cart2int`
RegSAHam::RegSAHam(const SAHamLoader & loader,
std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> (*cart2int)(const at::Tensor &))
: SAGeometry(loader.geom, loader.CNPI2point, cart2int),
energy_(loader.energy.clone()), dH_(loader.dH.clone()) {
    for (size_t i = 0    ; i < dH_.size(0); i++)
    for (size_t j = i + 1; j < dH_.size(1); j++)
    dH_[i][j] *= energy_[j] - energy_[i];
    this->construct_symmetry_();
    this->reconstruct_dH_();

    size_t NStates = energy_.size(0);
    weight_E_.resize(NStates);
    std::fill(weight_E_.begin(), weight_E_.end(), 1.0);
    sqrtweight_E_.resize(NStates);
    std::fill(sqrtweight_E_.begin(), sqrtweight_E_.end(), 1.0);
    weight_dH_.resize(NStates);
    weight_dH_ = 1.0;
    sqrtweight_dH_.resize(NStates);
    sqrtweight_dH_ = 1.0;
}
RegSAHam::~RegSAHam() {}

const at::Tensor & RegSAHam::energy() const {return energy_;}
const at::Tensor & RegSAHam::dH() const {return dH_;}
const CL::utility::matrix<size_t> & RegSAHam::irreds() const {return irreds_;}
const CL::utility::matrix<at::Tensor> & RegSAHam::SAdH() const {return SAdH_;}

size_t RegSAHam::NStates() const {return energy_.size(0);}
const double & RegSAHam::weight_E(const size_t & index) const {return weight_E_[index];}
const double & RegSAHam::sqrtweight_E(const size_t & index) const {return sqrtweight_E_[index];}
const double & RegSAHam::weight_dH(const size_t & row, const size_t & column) const {return weight_dH_[row][column];}
const double & RegSAHam::sqrtweight_dH(const size_t & row, const size_t & column) const {return sqrtweight_dH_[row][column];}

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
// Lower the weight if energy > E_thresh or ||dH|| > dH_thresh
void RegSAHam::adjust_weight(const double & E_thresh, const double & dH_thresh) {
    int64_t NStates = energy_.size(0);
    for (int64_t i = 0; i < NStates; i++) {
        double e = energy_[i].item<double>();
        if (e > E_thresh) {
            sqrtweight_E_[i] = E_thresh / e;
            weight_E_[i] = sqrtweight_E_[i] * sqrtweight_E_[i];
        }
    }
    for (int64_t i = 0; i < NStates; i++)
    for (int64_t j = i; j < NStates; j++) {
        double g = dH_[i][j].norm().item<double>();
        if (g > dH_thresh) {
            sqrtweight_dH_[i][j] = dH_thresh / g;
            weight_dH_[i][j] = sqrtweight_dH_[i][j] * sqrtweight_dH_[i][j];
        }
    }
}





DegSAHam::DegSAHam() {}
DegSAHam::DegSAHam(const DegSAHam & source) : RegSAHam(source),
H_(source.H_), weight_H_(source.weight_H_), sqrtweight_H_(source.sqrtweight_H_) {}
// See the base class constructor for details of `cart2int`
DegSAHam::DegSAHam(const SAHamLoader & loader,
std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> (*cart2int)(const at::Tensor &))
: RegSAHam(loader, cart2int) {
    // Transform H and ▽H to composite representation
    H_ = energy_.clone();
    tchem::chem::composite_representation_(H_, dH_);
    this->construct_symmetry_();
    // point group symmetry consistency check
    for (size_t i = 0    ; i < H_.size(0); i++)
    for (size_t j = i + 1; j < H_.size(1); j++)
    if (abs(H_[i][j].item<double>()) > 1e-12 && irreds_[i][j] != 0)
    std::cerr << "Warning: inconsistent irreducible between H and ▽H\n";

    size_t NStates = H_.size(0);
    weight_H_.resize(NStates);
    weight_H_ = 1.0;
    sqrtweight_H_.resize(NStates);
    sqrtweight_H_ = 1.0;
}
DegSAHam::~DegSAHam() {}

const at::Tensor & DegSAHam::H() const {return H_;};

const double & DegSAHam::weight_H(const size_t & row, const size_t & column) const {return weight_H_[row][column];}
const double & DegSAHam::sqrtweight_H(const size_t & row, const size_t & column) const {return sqrtweight_H_[row][column];}

void DegSAHam::to(const c10::DeviceType & device) {
    RegSAHam::to(device);
    H_.to(device);
}

// Subtract zero point from energy and H
void DegSAHam::subtract_ZeroPoint(const double & zero_point) {
    RegSAHam::subtract_ZeroPoint(zero_point);
    H_ -= zero_point * at::eye(H_.size(0), H_.options());
}
// Lower the weight if H > H_thresh or ||dH|| > dH_thresh
void DegSAHam::adjust_weight(const double & H_thresh, const double & dH_thresh) {
    RegSAHam::adjust_weight(H_thresh, dH_thresh);
    int64_t NStates = energy_.size(0);
    for (int64_t i = 0; i < NStates; i++)
    for (int64_t j = i; j < NStates; j++) {
        double h = H_[i][j].item<double>();
        if (h > H_thresh) {
            sqrtweight_H_[i][j] = H_thresh / h;
            weight_H_[i][j] = sqrtweight_H_[i][j] * sqrtweight_H_[i][j];
        }
    }
}

} // namespace abinitio