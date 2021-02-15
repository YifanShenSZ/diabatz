#include <abinitio/SAgeometry.hpp>

namespace abinitio {

// Construct `point2CNPI_` based on constructed `CNPI2point_`
void SAGeometry::construct_symmetry_() {
    assert(("`CNPI2point_` must have been constructed", ! CNPI2point_.empty()));
    std::vector<size_t> copy = CNPI2point_;
    size_t NPointIrreds = std::set<size_t>(copy.begin(), copy.end()).size();
    point2CNPI_.resize(NPointIrreds);
    for (size_t i = 0; i < NPointIrreds; i++)
    for (size_t j = 0; j < CNPI2point_.size(); j++)
    if (CNPI2point_[j] == i) point2CNPI_[i].push_back(j);
}
// Construct `S_`, `Ss_` and `sqrtSs_` based on constructed `Js_` and `point2CNPI_`
void SAGeometry::construct_metric_() {
    assert(("`Js_` must have been constructed", ! Js_.empty()));
    at::Tensor J = torch::cat(Js_);
    S_ = J.mm(J.transpose(0, 1));
    Ss_.resize(NPointIrreds());
    std::vector<at::Tensor> Js_point = cat(Js_);
    for (size_t i = 0; i < Ss_.size(); i++) {
        Ss_[i] = Js_point[i].mm(Js_point[i].transpose(0, 1));
        at::Tensor eigvals, eigvecs;
        std::tie(eigvals, eigvecs) = Ss_[i].symeig(true);
        eigvals = at::sqrt(eigvals);
        sqrtSs_[i] = eigvecs.mm(eigvals.diag().mm(eigvecs.transpose(0, 1)));
    }
    
}

SAGeometry::SAGeometry() {}
// `cart2int` takes in Cartesian coordinate,
// returns symmetry adapted internal coordinates and corresponding Jacobians
SAGeometry::SAGeometry(const at::Tensor & geom, const std::vector<size_t> & _CNPI2point,
std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> (*cart2int)(const at::Tensor &))
: CNPI2point_(_CNPI2point) {
    std::tie(qs_, Js_) = cart2int(geom);
    this->construct_symmetry_();
    this->construct_metric_();
}
// See the base constructor for details of `cart2int`
SAGeometry::SAGeometry(const SAGeomLoader & loader,
std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> (*cart2int)(const at::Tensor &))
: SAGeometry(loader.geom, loader.CNPI2point, cart2int) {}
SAGeometry::~SAGeometry() {}

std::vector<size_t> SAGeometry::CNPI2point() const {return CNPI2point_;}
std::vector<at::Tensor> SAGeometry::qs() const {return qs_;}
std::vector<at::Tensor> SAGeometry::Js() const {return Js_;}
std::vector<std::vector<size_t>> SAGeometry::point2CNPI() const {return point2CNPI_;}
at::Tensor SAGeometry::S() const {return S_;}
std::vector<at::Tensor> SAGeometry::Ss() const {return Ss_;}
std::vector<at::Tensor> SAGeometry::sqrtSs() const {return sqrtSs_;}

size_t SAGeometry::NPointIrreds() const {
    assert(("`point2CNPI_` must have been constructed", ! point2CNPI_.empty()));
    return point2CNPI_.size();
}

void SAGeometry::to(const c10::DeviceType & device) {
    for (at::Tensor & q : qs_) q.to(device);
    for (at::Tensor & J : Js_) J.to(device);
    S_.to(device);
    for (at::Tensor & S : Ss_) S.to(device);
}

// Concatenate CNPI group symmetry adapted tensors to point group symmetry adapted tensors
std::vector<at::Tensor> SAGeometry::cat(std::vector<at::Tensor> xs) const {
    assert(("`point2CNPI_` must have been constructed", ! point2CNPI_.empty()));
    assert(("Number of CNPI group symmetry adapted tensors must equal to CNPI group order", xs.size() == qs_.size()));
    std::vector<at::Tensor> ys(NPointIrreds());
    for (size_t i = 0; i < ys.size(); i++) {
        std::vector<at::Tensor> xmatches(point2CNPI_[i].size());
        for (size_t j = 0; j < point2CNPI_[i].size(); j++) xmatches[j] = xs[point2CNPI_[i][j]];
        ys[i] = torch::cat(xmatches);
    }
    return ys;
}

} // namespace abinitio