#ifndef abinitio_SAgeometry_hpp
#define abinitio_SAgeometry_hpp

#include <abinitio/geometry.hpp>
#include <abinitio/SAloader.hpp>

namespace abinitio {

// Nomenclature:
// Cartesian coordinate: r
// CNPI  group symmetry adapted internal coordinate: q
// point group symmetry adapted internal coordinate: Q
// Jacobian of internal coordinate over Cartesian coordiate: J
// metric of internal coordinate gradient: S = J . J^T
class SAGeometry : public Geometry {
    protected:
        // q dimension of each irreducible
        std::vector<size_t> CNPI_intdims_;
        // the CNPI group symmetry adapted blocks of q, Jqr, Jqr^T
        std::vector<at::Tensor> qs_, Jqrs_, JqrTs_;
        // concatenation of the CNPI group symmetry adapted blocks
        at::Tensor Jqr_, JqrT_, Sq_;

        // Q dimension of each irreducible
        std::vector<size_t> point_intdims_;
        // the point group symmetry adapted blocks of Q, JQr, JQr^T, C2Q (C2Q . ▽r = ▽Q), S, sqrt(S)
        std::vector<at::Tensor> Qs_, JQrs_, JQrTs_, C2Qs_, SQs_, sqrtSQs_;
        // concatenation of the point group symmetry adapted blocks
        at::Tensor JQr_, JQrT_;

        // mapping between CNPI group and point group
        // CNPI irreducible i becomes point irreducible CNPI2point[i]
        std::vector<size_t> CNPI2point_;
        // point irreducible i contains CNPI irreducibles point2CNPI[i]
        std::vector<std::vector<size_t>> point2CNPI_;
    public:
        SAGeometry();
        // `cart2CNPI` takes in r, returns q and corresponding J
        SAGeometry(const std::string & _path, const double & _weight, const at::Tensor & _geom,
                   const std::vector<size_t> & _CNPI2point, const std::vector<std::string> & point_defs,
                   std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> (*cart2CNPI)(const at::Tensor &));
        SAGeometry(const SAGeomLoader & loader,
                   std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> (*cart2CNPI)(const at::Tensor &));
        ~SAGeometry();

        const std::vector<size_t> & CNPI_intdims() const;
        const std::vector<at::Tensor> & qs   () const;
        const std::vector<at::Tensor> & Jqrs () const;
        const std::vector<at::Tensor> & JqrTs() const;
        const at::Tensor & qs   (const size_t & irred) const;
        const at::Tensor & Jqrs (const size_t & irred) const;
        const at::Tensor & JqrTs(const size_t & irred) const;
        const at::Tensor & Jqr () const;
        const at::Tensor & JqrT() const;
        const at::Tensor & Sq  () const;
        const std::vector<size_t> & point_intdims() const;
        const std::vector<at::Tensor> & Qs     () const;
        const std::vector<at::Tensor> & JQrs   () const;
        const std::vector<at::Tensor> & JQrTs  () const;
        const std::vector<at::Tensor> & C2Qs   () const;
        const std::vector<at::Tensor> & SQs    () const;
        const std::vector<at::Tensor> & sqrtSQs() const;
        const at::Tensor & Qs     (const size_t & irred) const;
        const at::Tensor & JQrs   (const size_t & irred) const;
        const at::Tensor & JQrTs  (const size_t & irred) const;
        const at::Tensor & C2Qs   (const size_t & irred) const;
        const at::Tensor & SQs    (const size_t & irred) const;
        const at::Tensor & sqrtSQs(const size_t & irred) const;
        const std::vector<size_t> & CNPI2point() const;
        const std::vector<std::vector<size_t>> & point2CNPI() const;

        // number of point group irreducibles
        size_t NPointIrreds() const;

        void to(const c10::DeviceType & device);

        // concatenate CNPI group symmetry adapted tensors to point group symmetry adapted tensors
        std::vector<at::Tensor> cat(const std::vector<at::Tensor> & xs, const int64_t & dim = 0) const;
        // split an internal coordinate tensor to CNPI group symmetry adapted tensors
        // `x` is assumed to be the concatenation of CNPI group symmetry adapted internal coordinate tensors
        std::vector<at::Tensor> split2CNPI(const at::Tensor & x, const int64_t & dim = 0) const;
        // split an internal coordinate tensor to point group symmetry adapted tensors
        // `x` is assumed to be the concatenation of point group symmetry adapted internal coordinate tensors
        std::vector<at::Tensor> split2point(const at::Tensor & x, const int64_t & dim = 0) const;
};

} // namespace abinitio

#endif