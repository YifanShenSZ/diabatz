#include <CppLibrary/argparse.hpp>
#include <CppLibrary/chemistry.hpp>

#include <tchem/linalg.hpp>

#include <Hd/kernel.hpp>

argparse::ArgumentParser parse_args(const size_t & argc, const char ** & argv) {
    CL::utility::echo_command(argc, argv, std::cout);
    std::cout << '\n';
    argparse::ArgumentParser parser("Hessian for diabatz");

    // required arguments
    parser.add_argument("-d","--diabatz", '+', false, "diabatz definition files");
    parser.add_argument("-x","--xyz",       1, false, "the xyz geometry to calculate Hessian");

    // optional argument
    parser.add_argument("-a","--adiabatz");

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

// here ddHa is ▽[(▽H)a], computed by finite difference of (▽H)a
at::Tensor compute_ddHa(const at::Tensor & r, const Hd::kernel & Hdkernel) {
    const double dr = 1e-3;
    std::vector<at::Tensor> plus(r.size(0)), minus(r.size(0));
    #pragma omp parallel for
    for (size_t i = 0; i < r.size(0); i++) {
        at::Tensor Hd, dHd, energy, states;
        plus[i] = r.clone();
        plus[i][i] += dr;
        std::tie(Hd, dHd) = Hdkernel.compute_Hd_dHd(plus[i]);
        std::tie(energy, states) = Hd.symeig(true);
        plus[i] = tchem::linalg::UT_sy_U(dHd, states);
        minus[i] = r.clone();
        minus[i][i] -= dr;
        std::tie(Hd, dHd) = Hdkernel.compute_Hd_dHd(minus[i]);
        std::tie(energy, states) = Hd.symeig(true);
        minus[i] = tchem::linalg::UT_sy_U(dHd, states);
    }
    at::Tensor ddHa = r.new_empty({plus[0].size(0), plus[0].size(1), r.size(0), r.size(0)});
    for (size_t i = 0; i < r.size(0); i++) ddHa.select(2, i).copy_((plus[i] - minus[i]) / 2.0 / dr);
    return ddHa;
}

void print_matrix(const at::Tensor & matrix, const std::string & file) {
    std::ofstream ofs; ofs.open(file);
    for (size_t i = 0; i < matrix.size(0); i++) {
        for (size_t j = 0; j < matrix.size(0); j++) ofs << std::setw(16) << std::scientific << std::setprecision(6) << matrix[i][j].item<double>();
        ofs << '\n';
    }
    ofs.close();
}

int main(size_t argc, const char ** argv) {
    std::cout << "Hessian for diabatz\n"
              << "Yifan Shen 2021\n\n";
    argparse::ArgumentParser args = parse_args(argc, argv);
    CL::utility::show_time(std::cout);
    std::cout << '\n';

    std::vector<std::string> diabatz_inputs = args.retrieve<std::vector<std::string>>("diabatz");
    Hd::kernel Hdkernel(diabatz_inputs);

    CL::chem::xyz<double> geom(args.retrieve<std::string>("structure"), true);
    std::vector<double> coords = geom.coords();
    at::Tensor r = at::from_blob(coords.data(), coords.size(), at::TensorOptions().dtype(torch::kFloat64));

    at::Tensor ddH;
    std::string prefix;
    if (args.gotArgument("adiabatz")) {
        ddH = compute_ddHa(r, Hdkernel);
        prefix = "adiabatic-Hessian-";
    }
    else {
        ddH = compute_ddHd(r, Hdkernel);
        prefix = "diabatic-Hessian-";
    }
    size_t NStates = Hdkernel.NStates();
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = i; j < NStates; j++)
    print_matrix(ddH[i][j], prefix + std::to_string(i + 1) + "-" + std::to_string(j + 1) + ".txt");

    CL::utility::show_time(std::cout);
    std::cout << "Mission success\n";
}