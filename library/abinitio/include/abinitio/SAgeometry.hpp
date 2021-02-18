#ifndef abinitio_SAgeometry_hpp
#define abinitio_SAgeometry_hpp

#include <abinitio/SAloader.hpp>

namespace abinitio {

class SAGeometry {
    protected:
        // mapping from CNPI group to point group
        // CNPI irreducible i becomes point irreducible CNPI2point[i]
        std::vector<size_t> CNPI2point_;
        // CNPI group symmetry adapted internal coordinates
        // and their Jacobians over Cartesian coordiate
        std::vector<at::Tensor> qs_, Js_;

        // point irreducible i contains CNPI irreducibles point2CNPI[i]
        std::vector<std::vector<size_t>> point2CNPI_;
        // the metric of internal coordinate gradient S = J . J^T
        at::Tensor S_;
        // the point group symmetry adapted blocks of S and sqrt(S)
        std::vector<at::Tensor> Ss_, sqrtSs_;

        // Construct `point2CNPI_` based on constructed `CNPI2point_`
        void construct_symmetry_();
        // Construct `S_`, `Ss_` and `sqrtSs_` based on constructed `Js_` and `point2CNPI_`
        void construct_metric_();
    public:
        SAGeometry();
        // `cart2int` takes in Cartesian coordinate,
        // returns CNPI group symmetry adapted internal coordinates and corresponding Jacobians
        SAGeometry(const at::Tensor & geom, const std::vector<size_t> & _CNPI2point,
                   std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> (*cart2int)(const at::Tensor &));
        // See the base constructor for details of `cart2int`
        SAGeometry(const SAGeomLoader & loader,
                   std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> (*cart2int)(const at::Tensor &));
        ~SAGeometry();

        std::vector<size_t> CNPI2point() const;
        std::vector<at::Tensor> qs() const;
        std::vector<at::Tensor> Js() const;
        std::vector<std::vector<size_t>> point2CNPI() const;
        at::Tensor S() const;
        std::vector<at::Tensor> Ss() const;
        std::vector<at::Tensor> sqrtSs() const;

        // Number of point group irreducibles
        size_t NPointIrreds() const;

        void to(const c10::DeviceType & device);

        // Concatenate CNPI group symmetry adapted tensors to point group symmetry adapted tensors
        std::vector<at::Tensor> cat(const std::vector<at::Tensor> & xs) const;
        // Split an internal coordinate tensor to CNPI group symmetry adapted tensors
        // x is assumed to be the concatenation of CNPI group symmetry adapted internal coordinate tensors
        std::vector<at::Tensor> split2CNPI(const at::Tensor & x) const;
        // Split an internal coordinate tensor to point group symmetry adapted tensors
        // x is assumed to be the concatenation of CNPI group symmetry adapted internal coordinate tensors
        std::vector<at::Tensor> split2point(const at::Tensor & x) const;
};

} // namespace abinitio

#endif