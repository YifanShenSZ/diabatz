#ifndef abinitio_geometry_hpp
#define abinitio_geometry_hpp

#include <abinitio/loader.hpp>

namespace abinitio {

class Geometry {
    protected:
        at::Tensor geom_;
    public:
        Geometry();
        Geometry(const GeomLoader & loader);
        ~Geometry();

        at::Tensor geom() const;

        void to(const c10::DeviceType & device);
};

} // namespace abinitio

#endif