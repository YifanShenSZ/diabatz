#include <tchem/chemistry.hpp>

void final2init(
const std::vector<size_t> & init_NModes, const std::vector<size_t> & final_NModes,
const at::Tensor & init_q, const at::Tensor & final_q,
const tchem::chem::SANormalMode & init_vib, const tchem::chem::SANormalMode & final_vib) {
    int64_t intdim = init_q.size(0);
    at::Tensor init_Linv = init_q.new_zeros({intdim, intdim});
    size_t start = 0;
    for (size_t i = 0; i < init_NModes.size(); i++) {
        size_t stop = start + init_NModes[i];
        init_Linv.slice(0, start, stop).slice(1, start, stop).copy_(init_vib.Linvs()[i]);
        start = stop;
    }
    at::Tensor final_L = final_q.new_zeros({intdim, intdim});
    start = 0;
    for (size_t i = 0; i < final_NModes.size(); i++) {
        size_t stop = start + final_NModes[i];
        final_L.slice(0, start, stop).slice(1, start, stop).copy_(final_vib.intmodes()[i]);
        start = stop;
    }
    final_L.transpose_(0, 1);
    at::Tensor T = init_Linv.mm(final_L),
               b = init_Linv.mv(final_q - init_q);
    // Output
    std::ofstream ofs;
    std::cout << "The transformation matrix from final-state to initial-state normal modes can be found in transformation-matrix.txt\n";
    ofs.open("transformation-matrix.txt"); {
        for (size_t i = 0; i < intdim; i++) {
            for (size_t j = 0; j < intdim; j++)
            ofs << std::setw(25) << std::scientific << std::setprecision(15) << T[i][j].item<double>();
            ofs << '\n';
        }
    }
    ofs.close();
    std::cout << "The shift vector from final-state to initial-state normal modes can be found in shift-vector.txt\n";
    ofs.open("shift-vector.txt"); {
        for (size_t i = 0; i < intdim; i++)
        ofs << std::setw(25) << std::scientific << std::setprecision(15) << b[i].item<double>() << '\n';
    }
    ofs.close();
}