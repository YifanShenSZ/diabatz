#include <tchem/intcoord.hpp>

at::Tensor read_Columbus(const std::string & hessian_file, const tchem::IC::IntCoordSet & intcoordset) {
    size_t intdim = intcoordset.size();
    at::Tensor inthess = at::empty({(int64_t)intdim, (int64_t)intdim}, c10::TensorOptions().dtype(torch::kFloat64));
    // Read Columbus internal coordinate Hessian
    std::ifstream ifs; ifs.open(hessian_file);
    for (size_t i = 0; i < intdim; i++) {
        for (size_t j = 0; j < intdim / 8; j++) {
            double dbletemp;
            ifs >> dbletemp; inthess[i][8 * j + 0] = dbletemp;
            ifs >> dbletemp; inthess[i][8 * j + 1] = dbletemp;
            ifs >> dbletemp; inthess[i][8 * j + 2] = dbletemp;
            ifs >> dbletemp; inthess[i][8 * j + 3] = dbletemp;
            ifs >> dbletemp; inthess[i][8 * j + 4] = dbletemp;
            ifs >> dbletemp; inthess[i][8 * j + 5] = dbletemp;
            ifs >> dbletemp; inthess[i][8 * j + 6] = dbletemp;
            ifs >> dbletemp; inthess[i][8 * j + 7] = dbletemp;
        }
        if (intdim % 8 != 0) {
            size_t remain = intdim % 8, start = (intdim / 8) * 8;
            for (size_t j = 0; j < remain; j++) {
                double dbletemp;
                ifs >> dbletemp; inthess[i][start + j] = dbletemp;
            }
        }
    }
    ifs.close();
    // Columbus internal coordinate routines use weird unit:
    //     energy in 10^-18 J, length in A (to be continued)
    // So we need to convert Columbus Hessian to atomic unit
    inthess /= 4.35974417; // 1 Hatree = 4.35974417 * 10^-18 J
    for (size_t i = 0; i < intdim; i++)
    if (intcoordset[i][0].second.type() == "stretching") {
        inthess.select(0, i) /= 1.8897261339212517;
        inthess.select(1, i) /= 1.8897261339212517;
    }
    return inthess;
}