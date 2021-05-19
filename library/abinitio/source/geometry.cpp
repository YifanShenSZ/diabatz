#include <abinitio/geometry.hpp>

namespace abinitio {

Geometry::Geometry() {}
Geometry::Geometry(const Geometry & source) : geom_(source.geom_.clone()) {}
Geometry::Geometry(const at::Tensor & _geom) : geom_(_geom.clone()) {}
Geometry::Geometry(const GeomLoader & loader) : geom_(loader.geom.clone()) {}
Geometry::~Geometry() {}

const at::Tensor & Geometry::geom() const {return geom_;}

size_t Geometry::cartdim() const {return geom_.numel();}

void Geometry::to(const c10::DeviceType & device) {geom_.to(device);}

} // namespace abinitio