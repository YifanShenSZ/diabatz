#include <CppLibrary/argparse.hpp>

#include <obnet/symat.hpp>

argparse::ArgumentParser parse_args(const size_t & argc, const char ** & argv) {
    CL::utility::echo_command(argc, argv, std::cout);
    std::cout << '\n';
    argparse::ArgumentParser parser("Generate random parameters then output in text form for diabatz version 0");

    // required arguments
    parser.add_argument("-n","--net",            1, false, "diabatic Hamiltonian network definition file");
    parser.add_argument("-l","--input_layers", '+', false, "network input layer definition files");

    parser.parse_args(argc, argv);
    return parser;
}

int main(size_t argc, const char ** argv) {
    std::cout << "Generate random parameters then output in text form for diabatz version 0\n"
              << "Yifan Shen 2022\n\n";
    argparse::ArgumentParser args = parse_args(argc, argv);
    CL::utility::show_time(std::cout);
    std::cout << '\n';

    auto Hdnet = std::make_shared<obnet::symat>(args.retrieve<std::string>("net"));

    auto input_layers = args.retrieve<std::vector<std::string>>("input_layers");
    auto pmat = Hdnet->parameters();
    size_t count = 0;
    for (size_t istate = 0     ; istate < Hdnet->NStates(); istate++)
    for (size_t jstate = istate; jstate < Hdnet->NStates(); jstate++) {
        std::string prefix = "parameters_" + std::to_string(istate + 1) + "-" + std::to_string(jstate + 1) + "_";
        const auto & ps = pmat[istate][jstate];
        // The 1st layer is interpretable
        std::ifstream ifs; ifs.open(input_layers[count]);
        std::ofstream ofs; ofs.open(prefix + "1.txt");
        const at::Tensor & A = ps[0];
        for (size_t i = 0; i < A.size(1); i++) {
            std::string line;
            std::getline(ifs, line);
            ofs << line << '\n';
            for (size_t j = 0; j < A.size(0); j++)
            ofs << std::setw(25) << std::scientific << std::setprecision(15) << A[j][i].item<double>();
            ofs << '\n';
        }
        if (Hdnet->irreds()[istate][jstate] == 0) {
            const at::Tensor & b = ps[1];
            ofs << "bias\n";
            for (size_t i = 0; i < b.size(0); i++)
            ofs << std::setw(25) << std::scientific << std::setprecision(15) << b[i].item<double>();
            ofs << '\n';
        }
        ofs.close();
        ifs.close();
        // The other layers are uninterpretable
        if (Hdnet->irreds()[istate][jstate] == 0) 
        for (size_t layer = 2; layer < ps.size(); layer += 2) {
            std::ofstream ofs; ofs.open(prefix + std::to_string(layer / 2 + 1) + ".txt");
            const at::Tensor & A = ps[layer];
            ofs << "transposed transformation matrix\n";
            for (size_t i = 0; i < A.size(1); i++) {
                for (size_t j = 0; j < A.size(0); j++)
                ofs << std::setw(25) << std::scientific << std::setprecision(15) << A[j][i].item<double>();
                ofs << '\n';
            }
            const at::Tensor & b = ps[layer + 1];
            ofs << "bias\n";
            for (size_t i = 0; i < b.size(0); i++)
            ofs << std::setw(25) << std::scientific << std::setprecision(15) << b[i].item<double>();
            ofs << '\n';
        }
        else
        for (size_t layer = 1; layer < ps.size(); layer++) {
            std::ofstream ofs; ofs.open(prefix + std::to_string(layer + 1) + ".txt");
            const at::Tensor & A = ps[layer];
            ofs << "transposed transformation matrix\n";
            for (size_t i = 0; i < A.size(1); i++) {
                for (size_t j = 0; j < A.size(0); j++)
                ofs << std::setw(25) << std::scientific << std::setprecision(15) << A[j][i].item<double>();
                ofs << '\n';
            }
        }
        // Prepare for the next loop
        count++;
    }

    std::cout << '\n';
    CL::utility::show_time(std::cout);
    std::cout << "Mission success\n";
}