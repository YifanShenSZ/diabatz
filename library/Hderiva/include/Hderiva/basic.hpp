#ifndef Hderiva_basic_hpp
#define Hderiva_basic_hpp

#include <torch/torch.h>

namespace Hderiva {

// This routine calculates the [Ax, M] term in differentiating Ax
// A must be a matrix or higher order tensor, with the first 2 indices as the matrix indices
// M must be a 3rd-order tensor, with the last index as the gradient index
// The result has the gradient index at the end
// A is  symmetric: A[i][j] =  A[j][i]
// M is asymmetric: M[i][j] = -M[j][i]
// So the result is symmetric
// Only read the "upper triangle" (i <= j) of A
// Only read the "strict upper triangle" (i < j) of M
// Only write the "upper triangle" (i <= j) of the output tensor
at::Tensor commutor_term(const at::Tensor & A, const at::Tensor & M);

} // namespace Hderiva

#endif