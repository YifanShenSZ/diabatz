#ifndef train_trust_region_common_hpp
#define train_trust_region_common_hpp

#include "../../../include/data_classes.hpp"

#include "../common.hpp"

namespace train { namespace trust_region {

// data set
extern std::vector<std::shared_ptr<RegHam>> regset;
extern std::vector<std::shared_ptr<DegHam>> degset;
extern std::vector<std::shared_ptr<Energy>> energy_set;

// Each thread owns a chunk of data
extern std::vector<std::vector<std::shared_ptr<RegHam>>> regchunk;
extern std::vector<std::vector<std::shared_ptr<DegHam>>> degchunk;
extern std::vector<std::vector<std::shared_ptr<Energy>>> energy_chunk;

// Each thread works on a segment of residue or Jacobian
// Thread i works on rows [segstart[i], segstart[i + 1])
extern std::vector<size_t> segstart;

} // namespace trust_region
} // namespace train

#endif