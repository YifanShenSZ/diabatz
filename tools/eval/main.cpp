#include <CppLibrary/argparse.hpp>
#include <CppLibrary/chemistry.hpp>

#include <tchem/linalg.hpp>

#include <Hd/kernel.hpp>

argparse::ArgumentParser parse_args(const size_t & argc, const char ** & argv) {
    CL::utility::echo_command(argc, argv, std::cout);
    std::cout << '\n';
    argparse::ArgumentParser parser("Evaluation for diabatz");

    // required arguments
    parser.add_argument("-d","--diabatz", '+', false, "diabatz definition files");
    parser.add_argument("-x","--xyz", 1, false, "the xyz geometry to calculate diabatz");

    // optional argument
    parser.add_argument("-a","--adiabatz", (char)0, true, "use adiabatic rather than diabatic representation");
    parser.add_argument("-g","--gradient", (char)0, true, "additionally output gradient");
    parser.add_argument("--diagnostic");

    parser.parse_args(argc, argv);
    return parser;
}

void print_vector(const at::Tensor & vector, std::ostream & ostream) {
    at::Tensor r = vector.view({vector.size(0) / 3, 3});
    for (size_t i = 0; i < vector.size(0) / 3; i++)
    ostream << std::setw(16) << std::scientific << std::setprecision(6) << r[i][0].item<double>()
            << std::setw(16) << std::scientific << std::setprecision(6) << r[i][1].item<double>()
            << std::setw(16) << std::scientific << std::setprecision(6) << r[i][2].item<double>()
            << '\n';
}

void print_matrix(const at::Tensor & matrix, std::ostream & ostream) {
    for (size_t i = 0; i < matrix.size(0); i++) {
        for (size_t j = 0; j < matrix.size(0); j++) ostream << std::setw(16) << std::scientific << std::setprecision(6) << matrix[i][j].item<double>();
        ostream << '\n';
    }
}

void print_symat(const at::Tensor & matrix, std::ostream & ostream) {
    for (size_t i = 0; i < matrix.size(0); i++) {
        for (size_t j = 0; j < i; j++) ostream << "                ";
        for (size_t j = i; j < matrix.size(0); j++) ostream << std::setw(16) << std::scientific << std::setprecision(6) << matrix[i][j].item<double>();
        ostream << '\n';
    }
}

int main(size_t argc, const char ** argv) {
    std::cout << "Evaluation for diabatz\n"
              << "Yifan Shen 2021\n\n";
    argparse::ArgumentParser args = parse_args(argc, argv);
    CL::utility::show_time(std::cout);
    std::cout << '\n';

    std::vector<std::string> diabatz_inputs = args.retrieve<std::vector<std::string>>("diabatz");
    Hd::kernel Hdkernel(diabatz_inputs);

    CL::chem::xyz<double> xyz(args.retrieve<std::string>("xyz"), true);
    std::vector<double> coords = xyz.coords();
    at::Tensor r = at::from_blob(coords.data(), coords.size(), at::TensorOptions().dtype(torch::kFloat64));

    if (args.gotArgument("adiabatz")) {
        if (args.gotArgument("gradient")) {
            at::Tensor Hd, dHd;
            std::tie(Hd, dHd) = Hdkernel.compute_Hd_dHd(r);
            at::Tensor energy, states;
            std::tie(energy, states) = Hd.symeig(true);
            // energy
            std::cout << "energy =\n";
            for (size_t i = 0; i < energy.size(0); i++)
            std::cout << std::setw(16) << std::scientific << std::setprecision(6) << energy[i].item<double>();
            std::cout << "\n\n";
            // states
            std::cout << "states are:\n";
            print_matrix(states, std::cout);
            std::cout << '\n';
            // energy gradient
            at::Tensor dHa = tchem::linalg::UT_sy_U(dHd, states);
            for (size_t i = 0; i < Hd.size(0); i++) {
                std::cout << "energy gradient of state " << i + 1 << " =\n";
                print_vector(dHa[i][i], std::cout);
                std::cout << '\n';
            }
            // nonadiabatic coupling
            for (size_t i = 0    ; i < Hd.size(0); i++)
            for (size_t j = i + 1; j < Hd.size(0); j++) {
                std::cout << "nonadiabatic coupling between state " << i + 1 << " and " << j + 1 << " =\n";
                print_vector(dHa[i][j] / (energy[j] - energy[i]), std::cout);
                std::cout << '\n';
            }
        }
        else {
            at::Tensor Hd = Hdkernel(r);
            at::Tensor energy, states;
            std::tie(energy, states) = Hd.symeig(true);
            // energy
            std::cout << "energy =\n";
            for (size_t i = 0; i < energy.size(0); i++)
            std::cout << std::setw(16) << std::scientific << std::setprecision(6) << energy[i].item<double>();
            std::cout << "\n\n";
            // states
            std::cout << "states are:\n";
            print_matrix(states, std::cout);
            std::cout << '\n';
        }
    }
    else {
        if (args.gotArgument("gradient")) {
            at::Tensor Hd, dHd;
            std::tie(Hd, dHd) = Hdkernel.compute_Hd_dHd(r);
            // Hd
            std::cout << "Hd =\n";
            print_symat(Hd, std::cout);
            std::cout << '\n';
            // ▽Hd
            for (size_t i = 0; i < Hd.size(0); i++)
            for (size_t j = i; j < Hd.size(0); j++) {
                std::cout << "▽Hd " << i + 1 << "-" << j + 1 << " =\n";
                print_vector(dHd[i][j], std::cout);
                std::cout << '\n';
            }
        }
        else {
            at::Tensor Hd = Hdkernel(r);
            // Hd
            std::cout << "Hd =\n";
            print_symat(Hd, std::cout);
            std::cout << '\n';
        }
    }

    if (args.gotArgument("diagnostic")) Hdkernel.diagnostic(r, std::cout);

    CL::utility::show_time(std::cout);
    std::cout << "Mission success\n";
}