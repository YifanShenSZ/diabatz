#include <CppLibrary/argparse.hpp>
#include <CppLibrary/utility.hpp>

#include <abinitio/reader.hpp>

#include "global.hpp"

void compare();

argparse::ArgumentParser parse_args(const size_t & argc, const char ** & argv) {
    CL::utility::echo_command(argc, argv, std::cout);
    std::cout << '\n';
    argparse::ArgumentParser parser("Root mean square deviation analyzer for diabatz version 0");

    // required arguments
    parser.add_argument("-f","--format",         1, false, "internal coordinate definition format (Columbus7, default)");
    parser.add_argument("-i","--IC",             1, false, "internal coordinate definition file");
    parser.add_argument("-s","--SAS",            1, false, "symmetry adaptation and scale definition file");
    parser.add_argument("-n","--net",            1, false, "diabatic Hamiltonian network definition file");
    parser.add_argument("-c","--checkpoint",     1, false, "a trained Hd parameter to continue with");
    parser.add_argument("-l","--input_layers", '+', false, "network input layer definition files");
    parser.add_argument("-d","--data",         '+', false, "data set list file or directory");

    // optional arguments
    parser.add_argument("-z","--zero_point", 1, true, "zero of potential energy, default = 0");

    parser.parse_args(argc, argv);
    return parser;
}

int main(size_t argc, const char ** argv) {
    std::cout << "Root mean square deviation analyzer for diabatz version 0: \n"
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

    Hdkernel = std::make_shared<Hd::kernel>(format, IC, SAS, net_in, chk, input_layers);

    std::vector<std::string> data = args.retrieve<std::vector<std::string>>("data");
    abinitio::Reader reader(data);
    std::shared_ptr<abinitio::DataSet<abinitio::RegHam>> temp_regset;
    std::shared_ptr<abinitio::DataSet<abinitio::DegHam>> temp_degset;
    std::tie(temp_regset, temp_degset) = reader.read_HamSet();
    regset = temp_regset->examples();
    degset = temp_degset->examples();
    double zero_point = 0.0;
    if (args.gotArgument("zero_point")) zero_point = args.retrieve<double>("zero_point");
    double weight = 1.0;
    if (args.gotArgument("weight")) weight = args.retrieve<double>("weight");
    for (auto & data : regset) data->subtract_ZeroPoint(zero_point);
    for (auto & data : degset) data->subtract_ZeroPoint(zero_point);

    compare();

    std::cout << '\n';
    CL::utility::show_time(std::cout);
    std::cout << "Mission success\n";
}