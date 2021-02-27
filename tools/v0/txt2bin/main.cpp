#include <CppLibrary/argparse.hpp>

#include <obnet/symat.hpp>

argparse::ArgumentParser parse_args(const size_t & argc, const char ** & argv) {
    CL::utility::echo_command(argc, argv, std::cout);
    std::cout << '\n';
    argparse::ArgumentParser parser("Convert binary fitting parameters to text for diabatz version 0");

    // required arguments
    parser.add_argument("-n","--net",          1, false, "diabatic Hamiltonian network definition file");
    parser.add_argument("-p","--parameters", '+', false, "text form Hd parameter files");

    // optional arguments
    parser.add_argument("-o","--output", 1, true, "the file to save Hd parameter, default = Hd.net");

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
    std::shared_ptr<obnet::symat> Hdnet = std::make_shared<obnet::symat>(net_in);

    std::vector<std::string> parameters = args.retrieve<std::vector<std::string>>("parameters");

    size_t count = 0;
    for (size_t istate = 0     ; istate < Hdnet->NStates(); istate++)
    for (size_t jstate = istate; jstate < Hdnet->NStates(); jstate++) {
        torch::NoGradGuard no_grad;
        auto ps = Hdnet->elements[count]->parameters();
        auto p = ps[0].view(ps[0].numel());
        std::ifstream ifs; ifs.open(parameters[count]);
        for (size_t i = 0; i < p.size(0); i++) {
            std::string line;
            std::getline(ifs, line);
            if (! ifs.good()) throw std::ios::eofbit;
            std::getline(ifs, line);
            if (! ifs.good()) throw std::ios::eofbit;
            p[i].fill_(std::stod(line));
        }
        if (ps.size() > 1) {
            std::string line;
            std::getline(ifs, line);
            if (! ifs.good()) throw std::ios::eofbit;
            std::getline(ifs, line);
            if (! ifs.good()) throw std::ios::eofbit;
            ps[1].fill_(std::stod(line));
        }
        ifs.close();
        count++;
    }

    std::string output = "Hd.net";
    if (args.gotArgument("output")) output = args.retrieve<std::string>("output");
    torch::save(Hdnet->elements, output);

    std::cout << '\n';
    CL::utility::show_time(std::cout);
    std::cout << "Mission success\n";
}