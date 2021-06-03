#ifndef abinitio_geometry_hpp
#define abinitio_geometry_hpp

#include <abinitio/loader.hpp>

namespace abinitio {

class Geometry {
    protected:
        double sqrtweight_ = 1.0, weight_ = 1.0;
        at::Tensor geom_;
    public:
        Geometry();
        Geometry(const Geometry & source);
        Geometry(const double & _weight, const at::Tensor & _geom);
        Geometry(const GeomLoader & loader);
        ~Geometry();

        const double & sqrtweight() const;
        const double & weight() const;
        const at::Tensor & geom() const;

        size_t cartdim() const;

        void to(const c10::DeviceType & device);
};

} // namespace abinitio

#endif