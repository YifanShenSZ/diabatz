#include <CppLibrary/argparse.hpp>
#include <CppLibrary/utility.hpp>

#include "../include/global.hpp"
#include "../include/data.hpp"
#include "../include/train.hpp"

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
    parser.add_argument("-w","--energy_weight", 1, true, "energy threshold in weight adjustment, default = 1");
    parser.add_argument("--gradient_weight",    1, true, "gradient threshold in weight adjustment, default = infer from energy threshold");
    parser.add_argument("-g","--guess_diag",  '+', true, "initial guess of Hd diagonal, default = pytorch initialization");
    parser.add_argument("-c","--checkpoint",    1, true, "a trained Hd parameter to continue from");

    // regularization arguments
    parser.add_argument("-r","--regularization", 1, true, "enable regularization and set strength, can be a scalar or a vector file");
    parser.add_argument("-p","--priors",       '+', true, "priors for regularization");

    // optimizer arguments
    parser.add_argument("-o","--optimizer",     1, true, "trust_region, Adam, SGD (default = trust_region)");
    parser.add_argument("-m","--max_iteration", 1, true, "default = 100");
    parser.add_argument("-b","--batch_size",    1, true, "for Adam or SGD, default = 32");
    parser.add_argument("--learning_rate",      1, true, "for Adam or SGD, default = 1e-3");
    parser.add_argument("--opt_chk",            1, true, "a checkpoint for Adam or SGD");

    parser.parse_args(argc, argv);
    return parser;
}

int main(size_t argc, const char ** argv) {
    std::cout << "Diabatz version 0\n"
              << "Yifan Shen 2021\n\n";
    argparse::ArgumentParser args = parse_args(argc, argv);
    CL::utility::show_time(std::cout);
    std::cout << '\n';

    std::string format = args.retrieve<std::string>("format"),
                IC     = args.retrieve<std::string>("IC"),
                SAS    = args.retrieve<std::string>("SAS");
    sasicset = std::make_shared<tchem::IC::SASICSet>(format, IC, SAS);

    Hdnet = std::make_shared<obnet::symat>(args.retrieve<std::string>("net"));
    Hdnet->train();
    if (args.gotArgument("guess_diag")) {
        auto guess_diag = args.retrieve<std::vector<double>>("guess_diag");
        if (guess_diag.size() != Hdnet->NStates()) throw std::invalid_argument(
        "argument guess_diag: inconsistent number of diagonal elements");
        auto ps = Hdnet->parameters();
        torch::NoGradGuard no_grad;
        for (size_t i = 0; i < Hdnet->NStates(); i++) {
            for (size_t j = 1; j < ps[i][i].size() - 1; j += 2) ps[i][i][j].fill_(0.0);
            ps[i][i].back().fill_(guess_diag[i]);
        }
    }
    if (args.gotArgument("checkpoint")) torch::load(Hdnet->elements, args.retrieve<std::string>("checkpoint"));

    std::vector<std::string> input_layers = args.retrieve<std::vector<std::string>>("input_layers");
    if (input_layers.size() != (Hdnet->NStates() + 1) * Hdnet->NStates() / 2) throw std::invalid_argument(
    "The number of input layers must match the number of Hd upper-triangle elements");
    input_generator = std::make_shared<InputGenerator>(Hdnet->NStates(), Hdnet->irreds(), input_layers, sasicset->NSASICs());

    std::vector<std::string> data = args.retrieve<std::vector<std::string>>("data");
    std::shared_ptr<abinitio::DataSet<RegHam>> regset;
    std::shared_ptr<abinitio::DataSet<DegHam>> degset;
    std::tie(regset, degset) = read_data(data);
    std::cout << "There are " << regset->size_int() << " data points in adiabatic representation\n"
              << "          " << degset->size_int() << " data points in composite representation\n\n";

    double zero_point = 0.0;
    if (args.gotArgument("zero_point")) zero_point = args.retrieve<double>("zero_point");
    for (const auto & example : regset->examples()) example->subtract_ZeroPoint(zero_point);
    for (const auto & example : degset->examples()) example->subtract_ZeroPoint(zero_point);

    double maxe = 0.0, maxg = 0.0;
    for (const auto & example : regset->examples()) {
        double temp = example->energy()[0].item<double>();
        maxe = temp > maxe ? temp : maxe;
        temp = example->dH()[0][0].abs().max().item<double>();
        maxg = temp > maxg ? temp : maxg;
    }
    std::cout << "maximum ground state energy = " << maxe << '\n'
              << "maximum ||ground state energy gradient||_infinity = " << maxg << '\n'; 
    if (maxe > 0.0) unit = maxg / maxe;
    else            unit = 1.0; // fail safe
    std::cout << "so we suggest to set gradient / energy scaling to around " << unit << "\n\n";
    unit_square = unit * unit;

    double H_weight = 1.0;
    if (args.gotArgument("energy_weight")) H_weight = args.retrieve<double>("energy_weight");
    double dH_weight = unit * H_weight;
    if (args.gotArgument("gradient_weight")) {
        dH_weight = args.retrieve<double>("gradient_weight");
        unit = dH_weight / H_weight;
        unit_square = unit * unit;
        std::cout << "According to user defined energy threshold and gradient threshold,\n"
                     "set gradient / energy scaling to " << unit << "\n\n";
    }
    for (const auto & example : regset->examples()) example->adjust_weight(H_weight, dH_weight);
    for (const auto & example : degset->examples()) example->adjust_weight(H_weight, dH_weight);

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

    train::initialize();
    size_t max_iteration = 100;
    if (args.gotArgument("max_iteration")) max_iteration = args.retrieve<size_t>("max_iteration");
    std::string optimizer = "trust_region";
    if (args.gotArgument("optimizer")) optimizer = args.retrieve<std::string>("optimizer");
    if (optimizer == "trust_region") {
        std::cout << "Optimizer is trust region\n\n";
        train::trust_region::initialize(regset, degset);
        train::trust_region::optimize(regularized, max_iteration);
    }
    else {
        size_t batch_size = 32;
        if (args.gotArgument("batch_size")) batch_size = args.retrieve<size_t>("batch_size");
        std::cout << "Set batch size to " << batch_size << '\n';
        double learning_rate = 1e-3;
        if (args.gotArgument("learning_rate")) learning_rate = args.retrieve<double>("learning_rate");
        std::cout << "Set learning rate to " << learning_rate << '\n';
        std::string opt_chk = "";
        if (args.gotArgument("opt_chk")) {
            opt_chk = args.retrieve<std::string>("opt_chk");
            std::cout << "Optimizer will continue from " << opt_chk << '\n';
        }
        if (optimizer == "Adam") {
            std::cout << "Optimizer is adaptive moment estimation (Adam)\n\n";
            train::torch_optim::Adam(regset, degset, max_iteration, batch_size, learning_rate, opt_chk);
        }
        else if (optimizer == "SGD") {
            std::cout << "Optimizer is stochastic gradient descent (SGD)\n\n";
            train::torch_optim::SGD(regset, degset, max_iteration, batch_size, learning_rate, opt_chk);
        }
        else throw std::invalid_argument("Unsupported optimizer " + optimizer);
    }

    std::cout << '\n';
    CL::utility::show_time(std::cout);
    std::cout << "Mission success\n";
}