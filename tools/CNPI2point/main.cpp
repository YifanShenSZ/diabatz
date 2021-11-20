#include <CppLibrary/argparse.hpp>
#include <CppLibrary/utility.hpp>

#include <abinitio/reader.hpp>

#include "global.hpp"

void map(const std::shared_ptr<abinitio::Geometry> & ham);

argparse::ArgumentParser parse_args(const size_t & argc, const char ** & argv) {
    CL::utility::echo_command(argc, argv, std::cout);
    std::cout << '\n';
    argparse::ArgumentParser parser("CNPI2point: A small tool to help to map the global CNPI group\n"
                                    "            to the local point group of a certain data point");

    // required arguments
    parser.add_argument("-f","--format", 1, false, "internal coordinate definition format (Columbus7, default)");
    parser.add_argument("-i","--IC",     1, false, "internal coordinate definition file");
    parser.add_argument("-s","--SAS",    1, false, "symmetry adaptation and scale definition file");
    parser.add_argument("-d","--data", '+', false, "data set list file or directory");

    // optional arguments
    parser.add_argument("-t","--threshold", 1, true, "threshold for zero (default = 1e-6)");

    parser.parse_args(argc, argv);
    return parser;
}

int main(size_t argc, const char ** argv) {
    std::cout << "CNPI2point: Map the global CNPI group to the local point group of a certain data point\n"
              << "Yifan Shen 2021\n\n";
    argparse::ArgumentParser args = parse_args(argc, argv);
    CL::utility::show_time(std::cout);
    std::cout << '\n';

    std::string format = args.retrieve<std::string>("format");
    std::string IC     = args.retrieve<std::string>("IC");
    std::string SAS    = args.retrieve<std::string>("SAS");
    sasicset = std::make_shared<SASDIC::SASDICSet>(format, IC, SAS);
    if (sasicset->NIrreds() == 0) {
        throw "Number of irreducibles is 0? Please check your symmetry adapted internal coordinate input";
    }
    else if (sasicset->NIrreds() == 1) {
        throw "No need to tag since there is no symmetry";
    }

    std::vector<std::string> data = args.retrieve<std::vector<std::string>>("data");
    abinitio::Reader reader(data);
    reader.pretty_print(std::cout);
    auto GeomSet = reader.read_GeomSet();

    if (args.gotArgument("threshold")) threshold = args.retrieve<double>("threshold");

    size_t count = 1;
    for (auto & example : GeomSet->examples()) {
        std::cout << "Geometry number " << count << '\n';
        map(example);
        std::cout << '\n';
        count++;
    }
}