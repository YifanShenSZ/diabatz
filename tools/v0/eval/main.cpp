#include <CppLibrary/argparse.hpp>
#include <CppLibrary/chemistry.hpp>

#include <tchem/linalg.hpp>

#include <Hd/kernel.hpp>

argparse::ArgumentParser parse_args(const size_t & argc, const char ** & argv) {
    CL::utility::echo_command(argc, argv, std::cout);
    std::cout << '\n';
    argparse::ArgumentParser parser("Evaluation for diabatz version 0");

    // required arguments
    parser.add_argument("-f","--format",         1, false, "internal coordinate definition format (Columbus7, default)");
    parser.add_argument("-i","--IC",             1, false, "internal coordinate definition file");
    parser.add_argument("-s","--SAS",            1, false, "symmetry adaptation and scale definition file");
    parser.add_argument("-n","--net",            1, false, "diabatic Hamiltonian network definition file");
    parser.add_argument("-c","--checkpoint",     1, false, "a trained Hd parameter to continue with");
    parser.add_argument("-l","--input_layers", '+', false, "network input layer definition files");
    parser.add_argument("-g","--geometry",       1, false, "the geometry to calculate diabatz");

    // optional argument
    parser.add_argument("-a","--adiabatz");

    parser.parse_args(argc, argv);
    return parser;
}

void print_grad(const at::Tensor & grad, std::ostream & ostream) {
    at::Tensor r = grad.view({grad.size(0) / 3, 3});
    for (size_t i = 0; i < grad.size(0) / 3; i++)
    ostream << r[i][0].item<double>() << ' '
            << r[i][1].item<double>() << ' '
            << r[i][2].item<double>() << '\n';
}

int main(size_t argc, const char ** argv) {
    std::cout << "Evaluation for diabatz version 0: \n"
              << "Yifan Shen 2021\n\n";
    argparse::ArgumentParser args = parse_args(argc, argv);
    CL::utility::show_time(std::cout);
    std::cout << '\n';

    std::string format = args.retrieve<std::string>("format");
    std::string IC     = args.retrieve<std::string>("IC");
    std::string SAS    = args.retrieve<std::string>("SAS");
    std::string net_in = args.retrieve<std::string>("net");
    std::string chk    = args.retrieve<std::string>("checkpoint");
    std::vector<std::string> input_layers = args.retrieve<std::vector<std::string>>("input_layers");
    
    Hd::kernel Hdkernel(format, IC, SAS, net_in, chk, input_layers);

    CL::chem::xyz<double> geom(args.retrieve<std::string>("geometry"), true);
    std::vector<double> coords = geom.coords();
    at::Tensor r = at::from_blob(coords.data(), coords.size(), at::TensorOptions().dtype(torch::kFloat64));
    at::Tensor Hd, dHd;
    std::tie(Hd, dHd) = Hdkernel.compute_Hd_dHd(r);

    if (args.gotArgument("adiabatz")) {
        at::Tensor energy, states;
        std::tie(energy, states) = Hd.symeig(true);
        std::cout << "energy =\n" << energy << '\n';
        at::Tensor dHa = tchem::linalg::UT_sy_U(dHd, states);
        std::cout << "energy gradients of\n";
        for (size_t i = 0; i < Hd.size(0); i++) {
            std::cout << "state " << i + 1 << ":\n";
            print_grad(dHa[i][i], std::cout);
        }
        std::cout << "nonadiabatic couplings between\n";
        for (size_t i = 0    ; i < Hd.size(0); i++)
        for (size_t j = i + 1; j < Hd.size(0); j++) {
            std::cout << "state " << i + 1 << " and " << j + 1 << ":\n";
            print_grad(dHa[i][j] / (energy[j] - energy[i]), std::cout);
        }
    }
    else {
        std::cout << "Hd =\n" << Hd << '\n';
        std::cout << "▽Hd =\n";
        for (size_t i = 0; i < Hd.size(0); i++)
        for (size_t j = i; j < Hd.size(0); j++) {
            std::cout << "state " << i + 1 << "-" << j + 1 << ":\n";
            print_grad(dHd[i][j], std::cout);
        }
    }

    std::cout << '\n';
    CL::utility::show_time(std::cout);
    std::cout << "Mission success\n";
}