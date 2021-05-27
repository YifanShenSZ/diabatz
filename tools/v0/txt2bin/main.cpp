#include <CppLibrary/argparse.hpp>

#include <obnet/symat.hpp>

argparse::ArgumentParser parse_args(const size_t & argc, const char ** & argv) {
    CL::utility::echo_command(argc, argv, std::cout);
    std::cout << '\n';
    argparse::ArgumentParser parser("Convert text fitting parameters to binary for diabatz version 0");

    // required arguments
    parser.add_argument("-n","--net",    1, false, "diabatic Hamiltonian network definition file");
    parser.add_argument("-p","--prefix", 1, false, "text form Hd parameter files are prefix_state1-state2_layer.txt");

    // optional arguments
    parser.add_argument("-o","--output", 1, true, "the file to save Hd parameter, default = Hd.net");

    parser.parse_args(argc, argv);
    return parser;
}

int main(size_t argc, const char ** argv) {
    std::cout << "Convert text fitting parameters to binary for diabatz version 0\n"
              << "Yifan Shen 2021\n\n";
    argparse::ArgumentParser args = parse_args(argc, argv);
    CL::utility::show_time(std::cout);
    std::cout << '\n';

    auto Hdnet = std::make_shared<obnet::symat>(args.retrieve<std::string>("net"));

    std::string prefix = args.retrieve<std::string>("prefix");

    auto pmat = Hdnet->parameters();
    for (size_t istate = 0     ; istate < Hdnet->NStates(); istate++)
    for (size_t jstate = istate; jstate < Hdnet->NStates(); jstate++) {
        torch::NoGradGuard no_grad;
        std::string prefix_now = prefix + "_" + std::to_string(istate + 1) + "-" + std::to_string(jstate + 1) + "_";
        const auto & ps = pmat[istate][jstate];
        // The 1st layer is interpretable
        std::string file = prefix_now + "1.txt";
        std::ifstream ifs; ifs.open(file);
        const at::Tensor & A = ps[0];
        for (size_t i = 0; i < A.size(1); i++) {
            std::string line;
            std::getline(ifs, line);
            if (! ifs.good()) throw CL::utility::file_error(file);
            std::getline(ifs, line);
            if (! ifs.good()) throw CL::utility::file_error(file);
            auto strs = CL::utility::split(line);
            if (strs.size() != A.size(0)) throw std::invalid_argument("inconsistent line");
            for (size_t j = 0; j < A.size(0); j++) A[j][i].fill_(std::stod(strs[j]));
        }
        if (Hdnet->irreds()[istate][jstate] == 0) {
            const at::Tensor & b = ps[1];
            std::string line;
            std::getline(ifs, line);
            if (! ifs.good()) throw CL::utility::file_error(file);
            std::getline(ifs, line);
            if (! ifs.good()) throw CL::utility::file_error(file);
            auto strs = CL::utility::split(line);
            if (strs.size() != b.size(0)) throw std::invalid_argument("inconsistent line");
            for (size_t j = 0; j < b.size(0); j++) b[j].fill_(std::stod(strs[j]));
        }
        ifs.close();
        // The other layers are uninterpretable
        if (Hdnet->irreds()[istate][jstate] == 0) 
        for (size_t layer = 2; layer < ps.size(); layer += 2) {
            std::string file = prefix_now + std::to_string(layer / 2 + 1) + ".txt";
            std::ifstream ifs; ifs.open(file);
            const at::Tensor & A = ps[layer];
            std::string line;
            std::getline(ifs, line);
            for (size_t i = 0; i < A.size(1); i++)
            for (size_t j = 0; j < A.size(0); j++) {
                double dbletemp;
                ifs >> dbletemp;
                if (! ifs.good()) throw CL::utility::file_error(file);
                A[j][i].fill_(dbletemp);
            }
            const at::Tensor & b = ps[layer + 1];
            ifs >> line;
            for (size_t i = 0; i < b.size(0); i++) {
                double dbletemp;
                ifs >> dbletemp;
                if (! ifs.good()) throw CL::utility::file_error(file);
                b[i].fill_(dbletemp);
            }
            ifs.close();
        }
        else
        for (size_t layer = 1; layer < ps.size(); layer++) {
            std::string file = prefix_now + std::to_string(layer + 1) + ".txt";
            std::ifstream ifs; ifs.open(file);
            const at::Tensor & A = ps[layer];
            std::string line;
            std::getline(ifs, line);
            for (size_t i = 0; i < A.size(1); i++)
            for (size_t j = 0; j < A.size(0); j++) {
                double dbletemp;
                ifs >> dbletemp;
                if (! ifs.good()) throw CL::utility::file_error(file);
                A[j][i].fill_(dbletemp);
            }
            ifs.close();
        }
    }

    std::string output = "Hd.net";
    if (args.gotArgument("output")) output = args.retrieve<std::string>("output");
    torch::save(Hdnet->elements, output);

    std::cout << '\n';
    CL::utility::show_time(std::cout);
    std::cout << "Mission success\n";
}