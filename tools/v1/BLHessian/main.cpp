#include <CppLibrary/argparse.hpp>
#include <CppLibrary/chemistry.hpp>

#include <tchem/linalg.hpp>

#include <Hd/kernel.hpp>

argparse::ArgumentParser parse_args(const size_t & argc, const char ** & argv) {
    CL::utility::echo_command(argc, argv, std::cout);
    std::cout << '\n';
    argparse::ArgumentParser parser("Bond-length Hessian from diabatz v0");

    // required arguments
    parser.add_argument("-d","--diabatz", '+', false, "diabatz v0 definition files");
    parser.add_argument("-x","--xyz",       1, false, "the xyz geometry to calculate bond-length Hessian");
    parser.add_argument("-f","--format",    1, false, "bond length definition format (Columbus7, default)");
    parser.add_argument("-i","--IC",        1, false, "bond length definition file");

    parser.parse_args(argc, argv);
    return parser;
}

// computed by finite difference of (▽H)d
at::Tensor compute_ddHd(const at::Tensor & r, const Hd::kernel & Hdkernel) {
    const double dr = 1e-3;
    std::vector<at::Tensor> plus(r.size(0)), minus(r.size(0));
    #pragma omp parallel for
    for (size_t i = 0; i < r.size(0); i++) {
        at::Tensor Hd, dHd;
        plus[i] = r.clone();
        plus[i][i] += dr;
        std::tie(Hd, plus[i]) = Hdkernel.compute_Hd_dHd(plus[i]);
        minus[i] = r.clone();
        minus[i][i] -= dr;
        std::tie(Hd, minus[i]) = Hdkernel.compute_Hd_dHd(minus[i]);
    }
    at::Tensor ddHd = r.new_empty({plus[0].size(0), plus[0].size(1), r.size(0), r.size(0)});
    for (size_t i = 0; i < r.size(0); i++) ddHd.select(2, i).copy_((plus[i] - minus[i]) / 2.0 / dr);
    return ddHd;
}

int main(size_t argc, const char ** argv) {
    std::cout << "Bond-length Hessian from diabatz v0\n"
              << "Yifan Shen 2021\n\n";
    argparse::ArgumentParser args = parse_args(argc, argv);
    CL::utility::show_time(std::cout);
    std::cout << '\n';

    std::vector<std::string> diabatz_inputs = args.retrieve<std::vector<std::string>>("diabatz");
    Hd::kernel Hdkernel(diabatz_inputs);

    CL::chem::xyz<double> xyz(args.retrieve<std::string>("structure"), true);
    std::vector<double> coords = xyz.coords();
    at::Tensor r = at::from_blob(coords.data(), coords.size(), at::TensorOptions().dtype(torch::kFloat64));

    std::string format = args.retrieve<std::string>("format"),
                IC     = args.retrieve<std::string>("IC");
    tchem::IC::IntCoordSet blset(format, IC);

    at::Tensor H, dH;
    std::tie(H, dH) = Hdkernel.compute_Hd_dHd(r);
    at::Tensor ddH = compute_ddHd(r, Hdkernel);

    for (size_t i = 0; i < Hdkernel.NStates(); i++) {
        at::Tensor cartgrad =  dH[i][i],
                   cartHess = ddH[i][i];
        at::Tensor intHess = blset.Hessian_cart2int(r, cartgrad, cartHess);
        std::ofstream ofs; ofs.open("BLHessian-" + std::to_string(i + 1) + ".txt");
        for (size_t j = 0; j < intHess.size(0); j++) ofs << intHess[j][j] << '\n';
        ofs.close();
    }

    CL::utility::show_time(std::cout);
    std::cout << "Mission success\n";
}