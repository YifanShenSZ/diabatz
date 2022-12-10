#ifndef abinitio_geometry_hpp
#define abinitio_geometry_hpp

#include <abinitio/loader.hpp>

namespace abinitio {

class Geometry {
    protected:
        std::string path_;
        double sqrtweight_ = 1.0, weight_ = 1.0;
        at::Tensor geom_;
    public:
        Geometry();
        Geometry(const std::string & _path, const double & _weight, const at::Tensor & _geom);
        Geometry(const GeomLoader & loader);
        ~Geometry();

        const std::string & path() const;
        const double & sqrtweight() const;
        const double & weight() const;
        const at::Tensor & geom() const;

        size_t cartdim() const;

        void set_weight(const double & _weight);
        void to(const c10::DeviceType & device);
};

} // namespace abinitio

#endif