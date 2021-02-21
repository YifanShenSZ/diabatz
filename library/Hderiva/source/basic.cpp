#include <tchem/linalg.hpp>

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
at::Tensor commutor_term(const at::Tensor & A, const at::Tensor & M) {
    assert(("A must be a matrix or higher order tensor", A.sizes().size() >= 2));
    assert(("M must be a 3rd-order tensor", M.sizes().size() == 3));
    assert(("The matrix part of A must be square", A.size(0) == A.size(1)));
    assert(("The matrix part of M must be square", M.size(0) == M.size(1)));
    assert(("A & M must be matrix mutiplicable", A.size(1) == M.size(0)));
    std::vector<int64_t> dims(A.sizes().size() + 1);
    for (size_t i = 0; i < A.sizes().size(); i++) dims[i] = A.size(i);
    dims.back() = M.size(-1);
    c10::IntArrayRef sizes(dims.data(), dims.size());
    at::Tensor result = A.new_zeros(sizes);
    size_t N = A.size(0);
    for (size_t i = 0; i < N; i++) {
        // k < i = j
        for (size_t k = 0; k < i; k++)
        result[i][i] += tchem::linalg::outer_product(A[k][i], M[k][i]);
        // i = j < k
        for (size_t k = i + 1; k < N; k++)
        result[i][i] -= tchem::linalg::outer_product(A[i][k], M[i][k]);
        result[i][i] *= 2.0;
        // i < j
        for (size_t j = i + 1; j < N; j++) {
            // k < i < j
            for (size_t k = 0; k < i; k++)
            result[i][j] += tchem::linalg::outer_product(A[k][i],  M[k][j])
                          - tchem::linalg::outer_product(A[k][j], -M[k][i]);
            // k = i < j
            result[i][j] += tchem::linalg::outer_product(A[i][i],  M[i][j]);
            // i < k < j, exist only if j - i >= 2
            for (size_t k = i + 1; k < j; k++)
            result[i][j] += tchem::linalg::outer_product(A[i][k], M[k][j])
                          - tchem::linalg::outer_product(A[k][j], M[i][k]);
            // i < j = k
            result[i][j] -= tchem::linalg::outer_product(A[j][j], M[i][j]);
            // i < j < k
            for (size_t k = j + 1; k < N; k++)
            result[i][j] += tchem::linalg::outer_product(A[i][k], -M[j][k])
                          - tchem::linalg::outer_product(A[j][k],  M[i][k]);
        }
    }
    return result;
}

} // namespace Hderiva