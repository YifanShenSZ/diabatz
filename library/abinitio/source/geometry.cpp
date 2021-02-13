#include <abinitio/geometry.hpp>

namespace abinitio {

Geometry::Geometry() {}
Geometry::Geometry(const GeomLoader & loader) {geom_ = loader.geom.clone();}
Geometry::~Geometry() {}

at::Tensor Geometry::geom() const {return geom_;}

void Geometry::to(const c10::DeviceType & device) {geom_.to(device);}

} // namespace abinitio