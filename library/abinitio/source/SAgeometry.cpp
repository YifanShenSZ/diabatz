#include <tchem/intcoord.hpp>

#include <abinitio/SAgeometry.hpp>

namespace abinitio {

SAGeometry::SAGeometry() {}

SAGeometry::SAGeometry(const SAGeometry & source) : Geometry(source),
CNPI_intdims_(source.CNPI_intdims_),
qs_(source.qs_), Jqrs_(source.Jqrs_), JqrTs_(source.JqrTs_),
Jqr_(source.Jqr_), JqrT_(source.JqrT_), Sq_(source.Sq_),
point_intdims_(source.point_intdims_),
Qs_(source.Qs_), JQrs_(source.JQrs_), JQrTs_(source.JQrTs_), C2Qs_(source.C2Qs_), SQs_(source.SQs_), sqrtSQs_(source.sqrtSQs_),
CNPI2point_(source.CNPI2point_), point2CNPI_(source.point2CNPI_) {}

// `cart2CNPI` takes in r, returns q and corresponding J
SAGeometry::SAGeometry(const double & _weight, const at::Tensor & _geom,
const std::vector<size_t> & _CNPI2point, const std::vector<std::string> & point_defs,
std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> (*cart2CNPI)(const at::Tensor &))
: Geometry(_weight, _geom), CNPI2point_(_CNPI2point) {
    // CNPI group symmetry adapted internal coordinate
    std::tie(qs_, Jqrs_) = cart2CNPI(_geom);
    size_t n_CNPI_irreds = qs_.size();
    CNPI_intdims_.resize(n_CNPI_irreds);
    JqrTs_       .resize(n_CNPI_irreds);
    for (size_t i = 0; i < n_CNPI_irreds; i++) {
        CNPI_intdims_[i] = qs_[i].size(0);
        JqrTs_       [i] = Jqrs_[i].transpose(0, 1);
    }
    Jqr_  = at::cat(Jqrs_);
    JqrT_ = Jqr_.transpose(0, 1);
    Sq_   = Jqr_.mm(JqrT_);
    // point group symmetry adapted internal coordinate
    size_t n_point_irreds = point_defs.size();
    point_intdims_.resize(n_point_irreds);
    Qs_           .resize(n_point_irreds);
    JQrs_         .resize(n_point_irreds);
    JQrTs_        .resize(n_point_irreds);
    C2Qs_         .resize(n_point_irreds);
    SQs_          .resize(n_point_irreds);
    sqrtSQs_      .resize(n_point_irreds);
    for (size_t i = 0; i < n_point_irreds; i++) {
        at::Tensor & Q = Qs_[i], & J = JQrs_[i], & S = SQs_[i];
        tchem::IC::IntCoordSet set("default", point_defs[i]);
        std::tie(Q, J) = set.compute_IC_J(_geom);
        point_intdims_[i] = Q.size(0);
        JQrTs_        [i] = J.transpose(0, 1);
        C2Qs_         [i] = set.gradient_cart2int_matrix(_geom);
        S = J.mm(J.transpose(0, 1));
        at::Tensor eigvals, eigvecs;
        std::tie(eigvals, eigvecs) = S.symeig(true);
        eigvals.sqrt_();
        sqrtSQs_[i] = eigvecs.mm(eigvals.diag().mm(eigvecs.transpose(0, 1)));
    }
    // mapping between CNPI group and point group
    std::vector<size_t> copy = CNPI2point_;
    point2CNPI_.resize(n_point_irreds);
    for (size_t i = 0; i < n_point_irreds; i++)
    for (size_t j = 0; j < CNPI2point_.size(); j++)
    if (CNPI2point_[j] == i) point2CNPI_[i].push_back(j);
}

SAGeometry::SAGeometry(const SAGeomLoader & loader,
std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> (*cart2CNPI)(const at::Tensor &))
: SAGeometry(loader.weight, loader.geom, loader.CNPI2point, loader.point_defs, cart2CNPI) {}

SAGeometry::~SAGeometry() {}

const std::vector<size_t> & SAGeometry::CNPI_intdims() const {return CNPI_intdims_;}
const std::vector<at::Tensor> & SAGeometry::qs   () const {return qs_   ;}
const std::vector<at::Tensor> & SAGeometry::Jqrs () const {return Jqrs_ ;}
const std::vector<at::Tensor> & SAGeometry::JqrTs() const {return JqrTs_;}
const at::Tensor & SAGeometry::qs   (const size_t & irred) const {return qs_   [irred];}
const at::Tensor & SAGeometry::Jqrs (const size_t & irred) const {return Jqrs_ [irred];}
const at::Tensor & SAGeometry::JqrTs(const size_t & irred) const {return JqrTs_[irred];}
const at::Tensor & SAGeometry::Jqr () const {return Jqr_ ;}
const at::Tensor & SAGeometry::JqrT() const {return JqrT_;}
const at::Tensor & SAGeometry::Sq  () const {return Sq_  ;}
const std::vector<size_t> & SAGeometry::point_intdims() const {return point_intdims_;}
const std::vector<at::Tensor> & SAGeometry::Qs     () const {return Qs_     ;}
const std::vector<at::Tensor> & SAGeometry::JQrs   () const {return JQrs_   ;}
const std::vector<at::Tensor> & SAGeometry::JQrTs  () const {return JQrTs_  ;}
const std::vector<at::Tensor> & SAGeometry::C2Qs   () const {return C2Qs_   ;}
const std::vector<at::Tensor> & SAGeometry::SQs    () const {return SQs_    ;}
const std::vector<at::Tensor> & SAGeometry::sqrtSQs() const {return sqrtSQs_;}
const at::Tensor & SAGeometry::Qs     (const size_t & irred) const {return Qs_     [irred];}
const at::Tensor & SAGeometry::JQrs   (const size_t & irred) const {return JQrs_   [irred];}
const at::Tensor & SAGeometry::JQrTs  (const size_t & irred) const {return JQrTs_  [irred];}
const at::Tensor & SAGeometry::C2Qs   (const size_t & irred) const {return C2Qs_   [irred];}
const at::Tensor & SAGeometry::SQs    (const size_t & irred) const {return SQs_    [irred];}
const at::Tensor & SAGeometry::sqrtSQs(const size_t & irred) const {return sqrtSQs_[irred];}
const std::vector<size_t> & SAGeometry::CNPI2point() const {return CNPI2point_;}
const std::vector<std::vector<size_t>> & SAGeometry::point2CNPI() const {return point2CNPI_;}

size_t SAGeometry::NPointIrreds() const {return point_intdims_.size();}

void SAGeometry::to(const c10::DeviceType & device) {
    this->Geometry::to(device);
    for (at::Tensor & q  : qs_   ) q .to(device);
    for (at::Tensor & J  : Jqrs_ ) J .to(device);
    for (at::Tensor & JT : JqrTs_) JT.to(device);
    Jqr_ .to(device);
    JqrT_.to(device);
    Sq_  .to(device);
    for (at::Tensor & Q     : Qs_     ) Q    .to(device);
    for (at::Tensor & C2Q   : C2Qs_   ) C2Q  .to(device);
    for (at::Tensor & S     : SQs_    ) S    .to(device);
    for (at::Tensor & sqrtS : sqrtSQs_) sqrtS.to(device);
}

// split an internal coordinate tensor to CNPI group symmetry adapted blocks
// `x` is assumed to be the concatenation of CNPI group symmetry adapted blocks
std::vector<at::Tensor> SAGeometry::split2CNPI(const at::Tensor & x, const int64_t & dim) const {
    std::vector<at::Tensor> xs(CNPI_intdims_.size());
    size_t start = 0, stop;
    for (size_t i = 0; i < xs.size(); i++) {
        size_t stop = start + CNPI_intdims_[i];
        xs[i] = x.slice(dim, start, stop);
        start = stop;
    }
    if (stop != x.size(dim)) throw std::invalid_argument(
    "abinitio::SAGeometry::split2CNPI: The target dimension of x must match the internal coordinate dimension");
    return xs;
}
// Split an internal coordinate tensor to point group symmetry adapted blocks
// `x` is assumed to be the concatenation of point group symmetry adapted blocks
std::vector<at::Tensor> SAGeometry::split2point(const at::Tensor & x, const int64_t & dim) const {
    std::vector<at::Tensor> xs(point_intdims_.size());
    size_t start = 0, stop;
    for (size_t i = 0; i < xs.size(); i++) {
        size_t stop = start + point_intdims_[i];
        xs[i] = x.slice(dim, start, stop);
        start = stop;
    }
    if (stop != x.size(dim)) throw std::invalid_argument(
    "abinitio::SAGeometry::split2point: The target dimension of x must match the internal coordinate dimension");
    return xs;
}

} // namespace abinitio