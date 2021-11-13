#ifndef SASIC_OthScalRul_hpp
#define SASIC_OthScalRul_hpp

#include <torch/torch.h>

namespace SASIC {

// The rule of internal coordinates who are scaled by others
// self is scaled by exp(-alpha * avg(scalers))
struct OthScalRul {
    size_t self;
    double alpha;
    at::Tensor scaler;

    OthScalRul();
    ~OthScalRul();
};

}

#endif