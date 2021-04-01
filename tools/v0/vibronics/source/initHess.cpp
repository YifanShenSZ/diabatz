#include <tchem/intcoord.hpp>

#include <abinitio/SAgeometry.hpp>

#include "cart2int.hpp"

at::Tensor read_Columbus(const at::Tensor & r, const std::string & hessian_file) {
    tchem::IC::IntCoordSet intcoordset("Columbus7", "intcfl");
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
    // internal coordinate -> Cartesian coordinate
    at::Tensor intgrad = inthess.new_zeros(intdim);
    at::Tensor carthess = intcoordset.Hessian_int2cart(r, intgrad, inthess);
    return carthess;
}

std::vector<at::Tensor> Hessian_cart2int(
const at::Tensor & r, const std::vector<size_t> & CNPI2point, const at::Tensor & carthess) {
    abinitio::SAGeometry init_SAgeom(r, CNPI2point, cart2int);
    // CNPI group symmetry adapted internal coordinate
    std::vector<at::Tensor> CNPI_qs, CNPI_Js;
    std::tie(CNPI_qs, CNPI_Js) = cart2int(r);
    // CNPI group -> point group
    std::vector<at::Tensor> qs = init_SAgeom.cat(CNPI_qs),
                            Js = init_SAgeom.cat(CNPI_Js);
    // Cartesian coordinate -> point group symmetry adapted internal coordinate
    std::vector<at::Tensor> Hs(qs.size());
    for (size_t i = 0; i < qs.size(); i++) {
        const at::Tensor & J = Js[i];
        at::Tensor JJT = J.mm(J.transpose(0, 1));
        at::Tensor cholesky = JJT.cholesky();
        at::Tensor inverse = at::cholesky_inverse(cholesky);
        at::Tensor AT = inverse.mm(J);
        at::Tensor A  = AT.transpose(0, 1);
        // initial-state equilibrium geometry, ||gradient|| = 0
        Hs[i] = AT.mm(carthess.mm(A));
    }
    return Hs;
}