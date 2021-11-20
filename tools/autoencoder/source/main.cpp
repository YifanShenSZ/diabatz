#include <CppLibrary/argparse.hpp>
#include <CppLibrary/utility.hpp>

#include <abinitio/SAreader.hpp>

#include "../include/global.hpp"
#include "../include/train.hpp"

argparse::ArgumentParser parse_args(const size_t & argc, const char ** & argv) {
    CL::utility::echo_command(argc, argv, std::cout);
    std::cout << '\n';
    argparse::ArgumentParser parser("Pretrain dimensionality reduction network by training an autoencoder");

    // required arguments
    parser.add_argument("-f","--format",      1, false, "internal coordinate definition format (Columbus7, default)");
    parser.add_argument("-i","--IC",          1, false, "internal coordinate definition file");
    parser.add_argument("-s","--SAS",         1, false, "symmetry adaptation and scale definition file");
    parser.add_argument("-I","--irreducible", 1, false, "the irreducible to autoencode");
    parser.add_argument("-n","--net",         2, false, "encoder and decoder networks definition files");
    parser.add_argument("-d","--data",      '+', false, "data set list file or directory");

    // optional arguments
    parser.add_argument("-c","--checkpoint",  '+', true, "a trained autoencoder parameter to continue with");
    parser.add_argument("-m","--max_iteration", 1, true, "default = 100");
    parser.add_argument("-o","--optimizer",     1, true, "Adam, SGD, trust_region (default = trust_region)");
    parser.add_argument("-b","--batch_size",    1, true, "batch size for Adam & SGD (default = 32)");
    parser.add_argument("-l","--learning_rate", 1, true, "learning rate for Adam & SGD (default = 1e-3)");

    parser.parse_args(argc, argv);
    return parser;
}

int main(size_t argc, const char ** argv) {
    std::cout << "Pretrain dimensionality reduction network by training an autoencoder\n"
              << "Yifan Shen 2021\n\n";
    argparse::ArgumentParser args = parse_args(argc, argv);
    CL::utility::show_time(std::cout);
    std::cout << '\n';

    auto format = args.retrieve<std::string>("format"),
         IC     = args.retrieve<std::string>("IC"),
         SAS    = args.retrieve<std::string>("SAS");
    sasicset = std::make_shared<SASDIC::SASDICSet>(format, IC, SAS);

    irreducible = args.retrieve<size_t>("irreducible") - 1;
    auto encoder_decoder = args.retrieve<std::vector<std::string>>("net");
    encoder = std::make_shared<DimRed::Encoder>(read_vector(encoder_decoder[0]), irreducible == 0);
    decoder = std::make_shared<DimRed::Decoder>(read_vector(encoder_decoder[1]), irreducible == 0);

    auto user_list = args.retrieve<std::vector<std::string>>("data");
    abinitio::SAReader reader(user_list, cart2int);
    reader.pretty_print(std::cout);
    geom_set = reader.read_SAGeomSet();
    std::cout << "There are " << geom_set->size_int() << " data points\n\n";

    size_t max_iteration = 100;
    if (args.gotArgument("max_iteration")) max_iteration = args.retrieve<size_t>("max_iteration");
    std::string optimizer = "trust_region";
    if (args.gotArgument("optimizer")) optimizer = args.retrieve<std::string>("optimizer");
    size_t batch_size = 32;
    if (args.gotArgument("batch_size")) batch_size = args.retrieve<size_t>("batch_size");
    double learning_rate = 1e-3;
    if (args.gotArgument("learning_rate")) learning_rate = args.retrieve<double>("learning_rate");

    if (optimizer == "Adam") Adam(max_iteration, batch_size, learning_rate);
    else if (optimizer == "SGD") SGD(max_iteration, batch_size, learning_rate);
    else trust_region(max_iteration);

    std::cout << '\n';
    CL::utility::show_time(std::cout);
    std::cout << "Mission success\n";
}