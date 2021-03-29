#include <CppLibrary/argparse.hpp>
#include <CppLibrary/utility.hpp>

#include "global.hpp"
#include "data.hpp"
#include "train.hpp"

argparse::ArgumentParser parse_args(const size_t & argc, const char ** & argv) {
    CL::utility::echo_command(argc, argv, std::cout);
    std::cout << '\n';
    argparse::ArgumentParser parser("Diabatz version 0");

    // required arguments
    parser.add_argument("-f","--format",         1, false, "internal coordinate definition format (Columbus7, default)");
    parser.add_argument("-i","--IC",             1, false, "internal coordinate definition file");
    parser.add_argument("-s","--SAS",            1, false, "symmetry adaptation and scale definition file");
    parser.add_argument("-n","--net",            1, false, "diabatic Hamiltonian network definition file");
    parser.add_argument("-l","--input_layers", '+', false, "network input layer definition files");
    parser.add_argument("-d","--data",         '+', false, "data set list file or directory");

    // optional arguments
    parser.add_argument("-z","--zero_point",    1, true, "zero of potential energy, default = 0");
    parser.add_argument("-w","--weight",        1, true, "Ethresh in weight adjustment, default = 1");
    parser.add_argument("-c","--checkpoint",    1, true, "a trained Hd parameter to continue with");
    parser.add_argument("-m","--max_iteration", 1, true, "default = 100");

    // regularization arguments
    parser.add_argument("-r","--regularization", 1, true, "enable regularization and set strength, can be a scalar or a vector file");
    parser.add_argument("-p","--priors",       '+', true, "priors for regularization");

    parser.parse_args(argc, argv);
    return parser;
}

int main(size_t argc, const char ** argv) {
    std::cout << "Diabatz version 0\n"
              << "Yifan Shen 2021\n\n";
    argparse::ArgumentParser args = parse_args(argc, argv);
    CL::utility::show_time(std::cout);
    std::cout << '\n';

    std::string format = args.retrieve<std::string>("format");
    std::string IC     = args.retrieve<std::string>("IC");
    std::string SAS    = args.retrieve<std::string>("SAS");
    sasicset = std::make_shared<tchem::IC::SASICSet>(format, IC, SAS);

    Hdnet = std::make_shared<obnet::symat>(args.retrieve<std::string>("net"));
    Hdnet->train();
    if (args.gotArgument("checkpoint")) torch::load(Hdnet->elements, args.retrieve<std::string>("checkpoint"));

    std::vector<std::string> input_layers = args.retrieve<std::vector<std::string>>("input_layers");
    if (input_layers.size() != (Hdnet->NStates() + 1) * Hdnet->NStates() / 2) throw std::invalid_argument(
    "The number of input layers must match the number of Hd upper-triangle elements");
    input_generator = std::make_shared<InputGenerator>(Hdnet->NStates(), Hdnet->irreds(), input_layers, sasicset->NSASICs());

    std::vector<std::string> data = args.retrieve<std::vector<std::string>>("data");
    double zero_point = 0.0;
    if (args.gotArgument("zero_point")) zero_point = args.retrieve<double>("zero_point");
    double weight = 1.0;
    if (args.gotArgument("weight")) weight = args.retrieve<double>("weight");
    std::shared_ptr<abinitio::DataSet<RegHam>> regset;
    std::shared_ptr<abinitio::DataSet<DegHam>> degset;
    std::tie(regset, degset) = read_data(data, zero_point, weight);
    std::cout << "There are " << regset->size_int() << " data points in adiabatic representation\n"
              << "          " << degset->size_int() << " data points in composite representation\n\n";

    bool regularized = args.gotArgument("regularization");
    if (regularized) {
        std::cout << "Got regularization strength, enable regularization\n\n";
        // Count the number of fitting parameters
        size_t NPars = 0;
        for (const at::Tensor & p : Hdnet->elements->parameters()) NPars += p.numel();
        // Get regularization strength
        regularization = Hdnet->elements->parameters()[0].new_empty(NPars);
        std::string regularization_input = args.retrieve<std::string>("regularization");
        std::ifstream ifs; ifs.open(regularization_input);
        if (ifs.good()) {
            size_t count = 0;
            while (true) {
                std::string line;
                std::getline(ifs, line);
                if (! ifs.good()) break;
                std::getline(ifs, line);
                if (! ifs.good()) break;
                double temp = std::stod(line);
                if (count >= NPars) throw std::invalid_argument(
                "Regularization strength and fitting parameter must share a same dimension");
                regularization[count] = temp;
                count++;
            }
            if (count != NPars) throw std::invalid_argument(
            "Regularization strength and fitting parameter must share a same dimension");
        }
        else {
            regularization.fill_(std::stod(regularization_input));
        }
        // Get prior
        prior = Hdnet->elements->parameters()[0].new_empty(NPars);
        if (! args.gotArgument("priors")) throw std::invalid_argument(
        "Priors must be provided for regularization");
        std::vector<std::string> priors_in = args.retrieve<std::vector<std::string>>("priors");
        if (priors_in.size() != (Hdnet->NStates() + 1) * Hdnet->NStates() / 2) throw std::invalid_argument(
        "The number of priors must match the number of Hd upper-triangle elements");
        size_t count_prior = 0, count_in = 0;
        for (size_t i = 0; i < Hdnet->NStates(); i++)
        for (size_t j = i; j < Hdnet->NStates(); j++) {
            std::ifstream ifs; ifs.open(priors_in[count_in]);
            if (! ifs.good()) throw CL::utility::file_error(priors_in[count_in]);
            while (true) {
                std::string line;
                std::getline(ifs, line);
                if (! ifs.good()) break;
                std::getline(ifs, line);
                if (! ifs.good()) break;
                double temp = std::stod(line);
                if (count_prior >= NPars) throw std::invalid_argument(
                "Prior and fitting parameter must share a same dimension");
                prior[count_prior] = temp;
                count_prior++;
            }
            ifs.close();
            count_in++;
        }
        if (count_prior != NPars) throw std::invalid_argument(
        "Prior and fitting parameter must share a same dimension");
    }

    size_t max_iteration = 100;
    if (args.gotArgument("max_iteration")) max_iteration = args.retrieve<size_t>("max_iteration");
    initialize(regset, degset);
    optimize(regularized, max_iteration);

    std::cout << '\n';
    CL::utility::show_time(std::cout);
    std::cout << "Mission success\n";
}