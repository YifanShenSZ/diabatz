#include <CppLibrary/argparse.hpp>
#include <CppLibrary/utility.hpp>

#include "global.hpp"
#include "data.hpp"

argparse::ArgumentParser parse_args(const size_t & argc, const char ** & argv) {
    CL::utility::echo_command(argc, argv, std::cout);
    std::cout << '\n';
    argparse::ArgumentParser parser("Bound diabatz: diabatic Hamiltonian generator for bound systems");

    // required arguments
    parser.add_argument("-f","--format",        1, false, "internal coordinate definition format (Columbus7, default)");
    parser.add_argument("-i","--IC",            1, false, "internal coordinate definition file");
    parser.add_argument("-s","--SAS",           1, false, "symmetry adaptation and scale definition file");
    parser.add_argument("-n","--net",           1, false, "diabatic Hamiltonian network definition file");
    parser.add_argument("-l","--input_layer", '+', false, "network input layer definition files");
    parser.add_argument("-d","--data",        '+', false, "data set list file or directory");

    // optional arguments
    parser.add_argument("-z","--zero_point",   1, true, "zero of potential energy, default = 0");
    parser.add_argument("-w","--weight",       1, true, "Ethresh in weight adjustment, default = 1");
    parser.add_argument("-g","--guess_diag", '+', true, "initial guess for Hd diagonal, default = pytorch initialization");

    parser.parse_args(argc, argv);
    return parser;
}

int main(size_t argc, const char ** argv) {
    std::cout << "Bound diabatz: diabatic Hamiltonian generator for bound systems\n"
              << "Yifan Shen 2021\n\n";
    argparse::ArgumentParser args = parse_args(argc, argv);
    CL::utility::show_time(std::cout);
    std::cout << '\n';

    std::string format = args.retrieve<std::string>("format");
    std::string IC     = args.retrieve<std::string>("IC");
    std::string SAS    = args.retrieve<std::string>("SAS");
    sasicset = std::make_shared<tchem::IC::SASICSet>(format, IC, SAS);

    Hdnet = std::make_shared<obnet::symat>(args.retrieve<std::string>("net"));
    Hdnet->to(torch::kFloat64);
    size_t NStates = Hdnet->NStates();

    std::vector<std::string> input_layer = args.retrieve<std::vector<std::string>>("input_layer");
    input_generator = std::make_shared<InputGenerator>(NStates, input_layer, sasicset->NSASICs());

    std::vector<std::string> data = args.retrieve<std::vector<std::string>>("data");
    double zero_point = 0.0;
    if (args.gotArgument("zero_point")) zero_point = args.retrieve<double>("zero_point");
    double weight = 1.0;
    if (args.gotArgument("weight")) weight = args.retrieve<double>("weight");
    std::shared_ptr<abinitio::DataSet<RegHam>> regset;
    std::shared_ptr<abinitio::DataSet<DegHam>> degset;
    std::tie(regset, degset) = read_data(data, zero_point, weight);
}