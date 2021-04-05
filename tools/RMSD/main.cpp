#include <CppLibrary/argparse.hpp>
#include <CppLibrary/utility.hpp>

#include <abinitio/reader.hpp>

#include "global.hpp"

void compare();

argparse::ArgumentParser parse_args(const size_t & argc, const char ** & argv) {
    CL::utility::echo_command(argc, argv, std::cout);
    std::cout << '\n';
    argparse::ArgumentParser parser("Root mean square deviation analyzer for diabatz");

    // required arguments
    parser.add_argument("-d","--diabatz",  '+', false, "diabatz definition files");
    parser.add_argument("-s","--data_set", '+', false, "data set list file or directory");

    // optional arguments
    parser.add_argument("-z","--zero_point", 1, true, "zero of potential energy, default = 0");

    parser.parse_args(argc, argv);
    return parser;
}

int main(size_t argc, const char ** argv) {
    std::cout << "Root mean square deviation analyzer for diabatz\n"
              << "Yifan Shen 2021\n\n";
    argparse::ArgumentParser args = parse_args(argc, argv);
    CL::utility::show_time(std::cout);
    std::cout << '\n';

    std::vector<std::string> diabatz_inputs = args.retrieve<std::vector<std::string>>("diabatz");
    Hdkernel = std::make_shared<Hd::kernel>(diabatz_inputs);

    std::vector<std::string> data = args.retrieve<std::vector<std::string>>("data_set");
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