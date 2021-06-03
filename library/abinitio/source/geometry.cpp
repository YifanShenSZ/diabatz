#include <abinitio/geometry.hpp>

namespace abinitio {

Geometry::Geometry() {}

Geometry::Geometry(const Geometry & source) : 
sqrtweight_(source.sqrtweight_), weight_(source.weight_),
geom_(source.geom_.clone()) {}

Geometry::Geometry(const double & _weight, const at::Tensor & _geom) :
sqrtweight_(sqrt(_weight)), weight_(_weight),
geom_(_geom.clone()) {}

Geometry::Geometry(const GeomLoader & loader) : Geometry(loader.weight, loader.geom) {}

Geometry::~Geometry() {}

const double & Geometry::sqrtweight() const {return sqrtweight_;}

const double & Geometry::weight() const {return weight_;}

const at::Tensor & Geometry::geom() const {return geom_;}

size_t Geometry::cartdim() const {return geom_.numel();}

void Geometry::to(const c10::DeviceType & device) {geom_.to(device);}

} // namespace abinitio