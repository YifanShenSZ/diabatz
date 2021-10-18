#ifndef DimRed_RedCoordSet_hpp
#define DimRed_RedCoordSet_hpp

#include <DimRed/encoder.hpp>

namespace DimRed {

// A set of reduced coordinates
// Nomenclature:
//     `r` is the output reduced coordinate
//     `x` is the input coordinate
//     `c` is the network parameters
struct RedCoordSet : torch::nn::Module {
    private:
        int64_t TotalIndim_, TotalReddim_, TotalNPars_;
        std::vector<int64_t> indims_, reddims_, NPars_;

        // Construct private sizes based on constructed networks `irreducibles`
        void construct_sizes_();
    public:
        torch::nn::ModuleList irreducibles;

        RedCoordSet();
        // This copy constructor performs a somewhat deepcopy,
        // where new modules are generated and have same values as `source`
        RedCoordSet(const std::shared_ptr<RedCoordSet> & source);
        RedCoordSet(const std::string & redcoordset_in);
        ~RedCoordSet();

        size_t NIrreds() const;

        at::Tensor cat_Jrx(const std::vector<at::Tensor> & Jrxs) const;
        at::Tensor cat_Jrc(const std::vector<at::Tensor> & Jrcs) const;
        at::Tensor cat_Krxc(const std::vector<at::Tensor> & Krxcs) const;

        std::vector<at::Tensor> forward(const std::vector<at::Tensor> & xs);
        std::vector<at::Tensor> operator()(const std::vector<at::Tensor> & xs);

        std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>>
        compute_r_Jrx(const std::vector<at::Tensor> & xs);

        std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>, std::vector<at::Tensor>, std::vector<at::Tensor>>
        compute_r_Jrx_Jrc_Krxc(const std::vector<at::Tensor> & xs);
};

} // namespace DimRed

#endif