#include "common.hpp"

namespace train { namespace trust_region {

// data set
std::vector<std::shared_ptr<RegHam>> regset;
std::vector<std::shared_ptr<DegHam>> degset;

// Each thread owns a chunk of data
std::vector<std::vector<std::shared_ptr<RegHam>>> regchunk;
std::vector<std::vector<std::shared_ptr<DegHam>>> degchunk;

// Each thread works on a segment of residue or Jacobian
// Thread i works on rows [segstart[i], segstart[i + 1])
std::vector<size_t> segstart;

} // namespace trust_region
} // namespace train