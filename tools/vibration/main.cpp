#include <CppLibrary/argparse.hpp>
#include <CppLibrary/chemistry.hpp>

#include <tchem/utility.hpp>
#include <tchem/intcoord.hpp>
#include <tchem/linalg.hpp>
#include <tchem/chemistry.hpp>

#include <Hd/Kernel.hpp>

argparse::ArgumentParser parse_args(const size_t & argc, const char ** & argv) {
    CL::utility::echo_command(argc, argv, std::cout);
    std::cout << '\n';
    argparse::ArgumentParser parser("Vibrational analysis for diabatz");

    // required arguments
    parser.add_argument("-f","--format",    1, false, "internal coordinate definition format (Columbus7, default)");
    parser.add_argument("-i","--IC",        1, false, "internal coordinate definition file");
    parser.add_argument("-t","--target",    1, false, "the target diabatic electronic state to analyze vibration, index starts from 1");
    parser.add_argument("-x","--xyz",       1, false, "the xyz geometry to analyze vibration");
    parser.add_argument("-m","--mass",      1, false, "the masses of atoms");
    parser.add_argument("-d","--diabatz", '+', false, "diabatz definition files");

    // optional arguments
    parser.add_argument("-a","--adiabatz", (char)0, true, "use adiabatic rather than diabatic representation");
    parser.add_argument("-o","--output",         1, true, "output file name (default = avogadro.log)");

    parser.parse_args(argc, argv);
    return parser;
}

// computed by finite difference of (▽H)d
at::Tensor compute_ddHd(const at::Tensor & r, const Hd::Kernel & HdKernel) {
    const double dr = 1e-3;
    std::vector<at::Tensor> plus(r.size(0)), minus(r.size(0));
    #pragma omp parallel for
    for (size_t i = 0; i < r.size(0); i++) {
        at::Tensor Hd, dHd;
        plus[i] = r.clone();
        plus[i][i] += dr;
        std::tie(Hd, plus[i]) = HdKernel.compute_Hd_dHd(plus[i]);
        minus[i] = r.clone();
        minus[i][i] -= dr;
        std::tie(Hd, minus[i]) = HdKernel.compute_Hd_dHd(minus[i]);
    }
    at::Tensor ddHd = r.new_empty({plus[0].size(0), plus[0].size(1), r.size(0), r.size(0)});
    for (size_t i = 0; i < r.size(0); i++) ddHd.select(2, i).copy_((plus[i] - minus[i]) / 2.0 / dr);
    return ddHd;
}

// here ddHa is ▽[(▽H)a], computed by finite difference of (▽H)a
at::Tensor compute_ddHa(const at::Tensor & r, const Hd::Kernel & HdKernel) {
    // Here ddHa is ▽[(▽H)a], computed by finite difference of (▽H)a
    const double dr = 1e-3;
    std::vector<at::Tensor> plus(r.size(0)), minus(r.size(0));
    #pragma omp parallel for
    for (size_t i = 0; i < r.size(0); i++) {
        at::Tensor Hd, dHd, energy, states;
        plus[i] = r.clone();
        plus[i][i] += dr;
        std::tie(Hd, dHd) = HdKernel.compute_Hd_dHd(plus[i]);
        std::tie(energy, states) = Hd.symeig(true);
        plus[i] = tchem::linalg::UT_sy_U(dHd, states);
        minus[i] = r.clone();
        minus[i][i] -= dr;
        std::tie(Hd, dHd) = HdKernel.compute_Hd_dHd(minus[i]);
        std::tie(energy, states) = Hd.symeig(true);
        minus[i] = tchem::linalg::UT_sy_U(dHd, states);
    }
    at::Tensor ddHa = r.new_empty({plus[0].size(0), plus[0].size(1), r.size(0), r.size(0)});
    for (size_t i = 0; i < r.size(0); i++) ddHa.select(2, i).copy_((plus[i] - minus[i]) / 2.0 / dr);
    return ddHa;
}

int main(size_t argc, const char ** argv) {
    std::cout << "Vibrational analysis for diabatz\n"
              << "Yifan Shen 2021\n\n";
    argparse::ArgumentParser args = parse_args(argc, argv);
    CL::utility::show_time(std::cout);
    std::cout << '\n';

    std::string format = args.retrieve<std::string>("format");
    std::string IC     = args.retrieve<std::string>("IC");
    tchem::IC::IntCoordSet intcoordset(format, IC);

    size_t target_state = args.retrieve<size_t>("target");
    std::cout << "The target electronic state is " << target_state << '\n';
    target_state -= 1;

    std::string  xyz_file = args.retrieve<std::string>("xyz"),
                mass_file = args.retrieve<std::string>("mass");
    CL::chem::xyz_mass<double> xyz(xyz_file, mass_file, true);

    std::vector<std::string> diabatz_inputs = args.retrieve<std::vector<std::string>>("diabatz");
    Hd::Kernel HdKernel(diabatz_inputs);

    std::vector<double> coords = xyz.coords();
    at::Tensor r = at::from_blob(coords.data(), coords.size(), at::TensorOptions().dtype(torch::kFloat64));

    at::Tensor Hd, dHd;
    std::tie(Hd, dHd) = HdKernel.compute_Hd_dHd(r);
    at::Tensor cartgrad, carthess;
    if (args.gotArgument("adiabatz")) {
        at::Tensor energy, states;
        std::tie(energy, states) = Hd.symeig(true);
        at::Tensor dHa = tchem::linalg::UT_sy_U(dHd, states);
        at::Tensor ddHa = compute_ddHa(r, HdKernel);
        cartgrad =  dHa[target_state][target_state];
        carthess = ddHa[target_state][target_state];
    }
    else {
        at::Tensor ddHd = compute_ddHd(r, HdKernel);
        cartgrad =  dHd[target_state][target_state];
        carthess = ddHd[target_state][target_state];
    }
    at::Tensor inthess = intcoordset.Hessian_cart2int(r, cartgrad, carthess);

    at::Tensor q, J;
    std::tie(q, J) = intcoordset.compute_IC_J(r);
    tchem::chem::IntNormalMode intvib(xyz.masses(), J, inthess);
    intvib.kernel();

    std::string output = "avogadro.log";
    if (args.gotArgument("output")) output = args.retrieve<std::string>("output");
    auto freqs = tchem::utility::tensor2vector(intvib.frequency());
    auto modes = tchem::utility::tensor2matrix(intvib.cartmode());
    CL::chem::xyz_vib<double> avogadro(xyz.symbols(), xyz.coords(), freqs, modes, true);
    avogadro.print(output);

    std::cout << '\n';
    CL::utility::show_time(std::cout);
    std::cout << "Mission success\n";
}