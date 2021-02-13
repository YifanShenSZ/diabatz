#ifndef abinitio_SAgeometry_hpp
#define abinitio_SAgeometry_hpp

#include <abinitio/SAloader.hpp>

namespace abinitio {

class SAGeometry {
    protected:
        // symmetry adapted internal coordinates
        std::vector<at::Tensor> qs_;

        // point group order
        size_t point_order_;
        // mapping from CNPI group to point group
        // CNPI irreducible i becomes point irreducible CNPI2point[i]
        std::vector<size_t> CNPI2point_;
        // i-th point irreducible contains CNPI irreducibles point2CNPI[i]
        std::vector<std::vector<size_t>> point2CNPI_;
    public:
        SAGeometry();
        SAGeometry(const SAGeomLoader & loader, std::vector<at::Tensor> (*cart2SAint)(const at::Tensor &));
        ~SAGeometry();

        std::vector<at::Tensor> qs() const;
        size_t point_order() const;
        std::vector<size_t> CNPI2point() const;
        std::vector<std::vector<size_t>> point2CNPI() const;

        void to(const c10::DeviceType & device);
};

} // namespace abinitio

#endif