#include <CppLibrary/chemistry.hpp>

#include <Hd/kernel.hpp>

void print_geom(const at::Tensor & geom, std::ostream & ostream) {
    at::Tensor r = geom.view({geom.size(0) / 3, 3});
    for (size_t i = 0; i < geom.size(0) / 3; i++)
    ostream << r[i][0].item<double>() << ' '
            << r[i][1].item<double>() << ' '
            << r[i][2].item<double>() << '\n';
}

int main() {
    c10::TensorOptions top = at::TensorOptions().dtype(torch::kFloat64);

    std::vector<std::string> sapoly_files = {"11.in", "12.in", "22.in"};
    Hd::kernel Hdkernel("default", "IntCoordDef", "SAS.in",
                        "Hd.in", "Hd.net",
                        sapoly_files);

    std::cout << "Cs minimum geometry:\n";
    CL::chem::xyz<double> geom1("min-Cs.xyz", true);
    std::vector<double> coords1 = geom1.coords();
    at::Tensor r1 = at::from_blob(coords1.data(), coords1.size(), top);
    at::Tensor Hd = Hdkernel(r1);
    std::cout << "Hd =\n" << Hd << '\n';
    at::Tensor dHd;
    std::tie(Hd, dHd) = Hdkernel.compute_Hd_dHd(r1);
    std::cout << "Hd =\n" << Hd << '\n';
    std::cout << "dHd =\n";
    for (size_t i = 0; i < Hd.size(0); i++)
    for (size_t j = i; j < Hd.size(0); j++) {
        std::cout << i << ' ' << j << ":\n";
        print_geom(dHd[i][j], std::cout);
    }

    std::cout << "C1 minimum geometry:\n";
    CL::chem::xyz<double> geom2("min-C1.xyz", true);
    std::vector<double> coords2 = geom2.coords();
    at::Tensor r2 = at::from_blob(coords2.data(), coords2.size(), top);
    Hd = Hdkernel(r2);
    std::cout << "Hd =\n" << Hd << '\n';
    std::tie(Hd, dHd) = Hdkernel.compute_Hd_dHd(r2);
    std::cout << "Hd =\n" << Hd << '\n';
    std::cout << "dHd =\n";
    for (size_t i = 0; i < Hd.size(0); i++)
    for (size_t j = i; j < Hd.size(0); j++) {
        std::cout << i << ' ' << j << ":\n";
        print_geom(dHd[i][j], std::cout);
    }
}