#include <abinitio/geometry.hpp>

namespace abinitio {

Geometry::Geometry() {}
Geometry::Geometry(const Geometry & source) : geom_(source.geom_) {}
Geometry::Geometry(const at::Tensor & _geom) : geom_(_geom.clone()) {}
Geometry::Geometry(const GeomLoader & loader) : Geometry(loader.geom) {}
Geometry::~Geometry() {}

at::Tensor Geometry::geom() const {return geom_;}

void Geometry::to(const c10::DeviceType & device) {geom_.to(device);}

} // namespace abinitio