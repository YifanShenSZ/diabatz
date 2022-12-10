#include <abinitio/geometry.hpp>

namespace abinitio {

Geometry::Geometry() {}
Geometry::Geometry(const std::string & _path, const double & _weight, const at::Tensor & _geom) :
path_(_path), sqrtweight_(sqrt(_weight)), weight_(_weight), geom_(_geom.clone()) {}
Geometry::Geometry(const GeomLoader & loader) : Geometry(loader.path, loader.weight, loader.geom) {}
Geometry::~Geometry() {}

const std::string & Geometry::path() const {return path_;}
const double & Geometry::sqrtweight() const {return sqrtweight_;}
const double & Geometry::weight() const {return weight_;}
const at::Tensor & Geometry::geom() const {return geom_;}

size_t Geometry::cartdim() const {return geom_.numel();}

void Geometry::set_weight(const double & _weight) {
    weight_ = _weight;
    sqrtweight_ = sqrt(_weight);
}
void Geometry::to(const c10::DeviceType & device) {geom_.to(device);}

} // namespace abinitio