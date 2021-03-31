#ifndef abinitio_SAgeometry_hpp
#define abinitio_SAgeometry_hpp

#include <abinitio/geometry.hpp>
#include <abinitio/SAloader.hpp>

namespace abinitio {

class SAGeometry : public Geometry {
    protected:
        // mapping from CNPI group to point group
        // CNPI irreducible i becomes point irreducible CNPI2point[i]
        std::vector<size_t> CNPI2point_;
        // Internal coordinate dimension
        size_t intdim_;
        // CNPI group symmetry adapted internal coordinate
        // and its Jacobian over Cartesian coordiate
        at::Tensor q_, Jqr_, JqrT_;
        // the metric of internal coordinate gradient S = J . J^T
        at::Tensor S_;

        // point irreducible i contains CNPI irreducibles point2CNPI[i]
        std::vector<std::vector<size_t>> point2CNPI_;
        // the CNPI group symmetry adapted blocks of q, Jqr, Jqr^T
        std::vector<at::Tensor> qs_, Jqrs_, JqrTs_;
        // the point group symmetry adapted blocks of S and sqrt(S)
        std::vector<at::Tensor> Ss_, sqrtSs_;

        // Construct `point2CNPI_` based on constructed `CNPI2point_`
        // Construct `Ss_` and `sqrtSs_` based on constructed `Jqrs_` and `JqrTs_`
        void construct_symmetry_();
    public:
        SAGeometry();
        SAGeometry(const SAGeometry & source);
        // `cart2int` takes in Cartesian coordinate,
        // returns CNPI group symmetry adapted internal coordinates and corresponding Jacobians
        SAGeometry(const at::Tensor & _geom, const std::vector<size_t> & _CNPI2point,
                   std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> (*cart2int)(const at::Tensor &));
        // See the base constructor for details of `cart2int`
        SAGeometry(const SAGeomLoader & loader,
                   std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> (*cart2int)(const at::Tensor &));
        ~SAGeometry();

        const std::vector<size_t> & CNPI2point() const;
        const size_t & intdim() const;
        const at::Tensor & q   () const;
        const at::Tensor & Jqr () const;
        const at::Tensor & JqrT() const;
        const at::Tensor & S   () const;
        const std::vector<std::vector<size_t>> & point2CNPI() const;
        const std::vector<at::Tensor> & qs    () const;
        const std::vector<at::Tensor> & Jqrs  () const;
        const std::vector<at::Tensor> & JqrTs () const;
        const std::vector<at::Tensor> & Ss    () const;
        const std::vector<at::Tensor> & sqrtSs() const;

        // Number of point group irreducibles
        size_t NPointIrreds() const;

        void to(const c10::DeviceType & device);

        // Concatenate CNPI group symmetry adapted tensors to point group symmetry adapted tensors
        std::vector<at::Tensor> cat(const std::vector<at::Tensor> & xs, const int64_t & dim = 0) const;
        // Split an internal coordinate tensor to CNPI group symmetry adapted tensors
        // `x` is assumed to be the concatenation of CNPI group symmetry adapted internal coordinate tensors
        std::vector<at::Tensor> split2CNPI(const at::Tensor & x, const int64_t & dim = 0) const;
        // Split an internal coordinate tensor to point group symmetry adapted tensors
        // `x` is assumed to be the concatenation of point group symmetry adapted internal coordinate tensors
        std::vector<at::Tensor> split2point(const at::Tensor & x, const int64_t & dim = 0) const;
};

} // namespace abinitio

#endif