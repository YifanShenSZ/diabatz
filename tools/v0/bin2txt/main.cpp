#include <CppLibrary/argparse.hpp>

#include <obnet/symat.hpp>

argparse::ArgumentParser parse_args(const size_t & argc, const char ** & argv) {
    CL::utility::echo_command(argc, argv, std::cout);
    std::cout << '\n';
    argparse::ArgumentParser parser("Convert binary fitting parameters to text for diabatz version 0");

    // required arguments
    parser.add_argument("-n","--net",            1, false, "diabatic Hamiltonian network definition file");
    parser.add_argument("-c","--checkpoint",     1, false, "a trained Hd parameter to continue with");
    parser.add_argument("-l","--input_layers", '+', false, "network input layer definition files");

    parser.parse_args(argc, argv);
    return parser;
}

int main(size_t argc, const char ** argv) {
    std::cout << "Convert binary fitting parameters to text for diabatz version 0\n"
              << "Yifan Shen 2021\n\n";
    argparse::ArgumentParser args = parse_args(argc, argv);
    CL::utility::show_time(std::cout);
    std::cout << '\n';

    std::string net_in = args.retrieve<std::string>("net");
    std::string chk    = args.retrieve<std::string>("checkpoint");
    std::shared_ptr<obnet::symat> Hdnet = std::make_shared<obnet::symat>(net_in);
    torch::load(Hdnet->elements, chk);

    std::vector<std::string> input_layers = args.retrieve<std::vector<std::string>>("input_layers");
    size_t count = 0;
    for (size_t istate = 0     ; istate < Hdnet->NStates(); istate++)
    for (size_t jstate = istate; jstate < Hdnet->NStates(); jstate++) {
        auto ps = Hdnet->elements[count]->parameters();
        auto p = ps[0].view(ps[0].numel());
        std::ifstream ifs; ifs.open(input_layers[count]);
        std::ofstream ofs; ofs.open("parameters_" + std::to_string(istate + 1) + "-" + std::to_string(jstate + 1) + ".txt");
        for (size_t i = 0; i < p.size(0); i++) {
            std::string line;
            std::getline(ifs, line);
            ofs << line << '\n'
                << p[i].item<double>() << '\n';
        }
        if (ps.size() > 1) ofs << "bias\n" << ps[1].item<double>() << '\n';
        ofs.close();
        ifs.close();
        count++;
    }

    std::cout << '\n';
    CL::utility::show_time(std::cout);
    std::cout << "Mission success\n";
}