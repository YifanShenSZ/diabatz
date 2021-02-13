#include <abinitio/SAgeometry.hpp>

namespace abinitio {

SAGeometry::SAGeometry() {}
SAGeometry::SAGeometry(const SAGeomLoader & loader, std::vector<at::Tensor> (*cart2SAint)(const at::Tensor &)) {
    qs_ = cart2SAint(loader.geom);
    CNPI2point_ = loader.CNPI2point;
    std::vector<size_t> copy = CNPI2point_;
    point_order_ = std::set<size_t>(copy.begin(), copy.end()).size();
    point2CNPI_.resize(point_order_);
    for (size_t i = 0; i < point_order_; i++)
    for (size_t j = 0; j < CNPI2point_.size(); j++)
    if (CNPI2point_[j] == i) point2CNPI_[i].push_back(j);
}
SAGeometry::~SAGeometry() {}

std::vector<at::Tensor> SAGeometry::qs() const {return qs_;}
size_t SAGeometry::point_order() const {return point_order_;}
std::vector<size_t> SAGeometry::CNPI2point() const {return CNPI2point_;}
std::vector<std::vector<size_t>> SAGeometry::point2CNPI() const {return point2CNPI_;}

void SAGeometry::to(const c10::DeviceType & device) {
    for (at::Tensor & q : qs_) q.to(device);
}

} // namespace abinitio