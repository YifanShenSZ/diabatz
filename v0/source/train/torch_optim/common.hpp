#ifndef train_torch_optim_common_hpp
#define train_torch_optim_common_hpp

#include "../../../include/data.hpp"

namespace train { namespace torch_optim {

at::Tensor reg_residue(const std::vector<std::shared_ptr<RegHam>> & batch);

at::Tensor deg_residue(const std::vector<std::shared_ptr<DegHam>> & batch);

at::Tensor reg_gradient(const std::vector<std::shared_ptr<RegHam>> & batch);

at::Tensor deg_gradient(const std::vector<std::shared_ptr<DegHam>> & batch);

} // namespace torch_optim
} // namespace train

#endif