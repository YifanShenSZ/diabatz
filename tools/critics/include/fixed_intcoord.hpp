#ifndef fixed_intcoord_hpp
#define fixed_intcoord_hpp

#include <torch/torch.h>

class Fixed_intcoord {
    private:
        int64_t intdim_;
        std::vector<size_t> fixed_coords_;

        std::vector<size_t> free_coords_;
        std::vector<double> fixed_values_;
    public:
        Fixed_intcoord();
        Fixed_intcoord(const int64_t & _intdim, const std::vector<size_t> & _fixed_coords, const at::Tensor & init_q);
        ~Fixed_intcoord();

        at::Tensor vector_free2total(const at::Tensor & V_free) const;
        at::Tensor vector_total2free(const at::Tensor & V) const;

        at::Tensor matrix_total2free(const at::Tensor & M) const;
};

#endif