#ifndef abinitio_geometry_hpp
#define abinitio_geometry_hpp

#include <abinitio/loader.hpp>

namespace abinitio {

class Geometry {
    protected:
        at::Tensor geom_;
    public:
        Geometry();
        Geometry(const Geometry & source);
        Geometry(const at::Tensor & _geom);
        Geometry(const GeomLoader & loader);
        ~Geometry();

        at::Tensor geom() const;

        size_t cartdim() const;

        void to(const c10::DeviceType & device);
};

} // namespace abinitio

#endif