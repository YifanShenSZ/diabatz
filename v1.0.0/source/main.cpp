#include <CppLibrary/argparse.hpp>
#include <CppLibrary/utility.hpp>

#include "../include/global.hpp"
#include "../include/data.hpp"
#include "../include/train.hpp"
#include "../include/utility.hpp"

argparse::ArgumentParser parse_args(const size_t & argc, const char ** & argv) {
    CL::utility::echo_command(argc, argv, std::cout);
    std::cout << '\n';
    argparse::ArgumentParser parser("Diabatz version 1.0.0");

    // required arguments
    parser.add_argument("--pretrained",        '+', false, "pretrained Hd network definition files");
    parser.add_argument("-f","--format",         1, false, "internal coordinate definition format (Columbus7, default)");
    parser.add_argument("-i","--IC",             1, false, "internal coordinate definition file");
    parser.add_argument("-s","--SAS",            1, false, "symmetry adaptation and scale definition file");
    parser.add_argument("-n","--net",            1, false, "diabatic Hamiltonian network definition file");
    parser.add_argument("-l","--input_layers", '+', false, "network input layer definition files");
    parser.add_argument("-d","--data",         '+', false, "data set list files or directories");

    // optional arguments
    parser.add_argument("--energy_data",      '+', true, "data set list files or directories without gradient");
    parser.add_argument("-z","--zero_point",    1, true, "zero of potential energy, default = 0");
    parser.add_argument("--energy_weight"  ,  '+', true, "energy (reference, threshold) for each state in weight adjustment, default = (0, 1)");
    parser.add_argument("--gradient_weight",    1, true, "gradient threshold in weight adjustment, default = infer from energy threshold");
    parser.add_argument("-c","--checkpoint",    1, true, "a trained Hd parameter to continue from");

    // regularization arguments
    parser.add_argument("-r","--regularization", 1, true, "enable regularization and set strength, can be a scalar or files regularization_state1-state2_layer.txt");
    parser.add_argument("-p","--prior",          1, true, "prior for regularization, can be a scalar or files prior_state1-state2_layer.txt, default = 0");

    // optimizer arguments
    parser.add_argument("-m","--max_iteration", 1, true, "default = 20");

    parser.parse_args(argc, argv);
    return parser;
}

int main(size_t argc, const char ** argv) {
    std::cout << "Diabatz version 1.0.0\n"
              << "Yifan Shen 2021\n\n";
    argparse::ArgumentParser args = parse_args(argc, argv);
    CL::utility::show_time(std::cout);
    std::cout << '\n';

    std::vector<std::string> pretrained_inputs = args.retrieve<std::vector<std::string>>("pretrained");
    pretrained_Hdkernel = std::make_shared<Hd::kernel>(pretrained_inputs);

    std::string format = args.retrieve<std::string>("format"),
                IC     = args.retrieve<std::string>("IC"),
                SAS    = args.retrieve<std::string>("SAS");
    sasicset = std::make_shared<SASDIC::SASDICSet>(format, IC, SAS);

    Hdnet = std::make_shared<obnet::symat>(args.retrieve<std::string>("net"));
    Hdnet->train();
    // since pretrained Hd should already have bias, set all output biases in Hd to 0
    for (int64_t istate = 0     ; istate < Hdnet->NStates(); istate++)
    for (int64_t jstate = istate; jstate < Hdnet->NStates(); jstate++)
    if (Hdnet->irreds()[istate][jstate] == 0) {
        torch::NoGradGuard no_grad;
        auto ps = Hdnet->parameters()[istate][jstate];
        ps[ps.size() - 1].fill_(0.0);
    }
    if (args.gotArgument("checkpoint")) torch::load(Hdnet->elements, args.retrieve<std::string>("checkpoint"));

    std::vector<std::string> input_layers = args.retrieve<std::vector<std::string>>("input_layers");
    if (input_layers.size() != (Hdnet->NStates() + 1) * Hdnet->NStates() / 2) throw std::invalid_argument(
    "The number of input layers must match the number of Hd upper-triangle elements");
    input_generator = std::make_shared<InputGenerator>(Hdnet->NStates(), Hdnet->irreds(), input_layers, sasicset->NSASDICs());

    std::vector<std::string> data = args.retrieve<std::vector<std::string>>("data");
    std::shared_ptr<abinitio::DataSet<RegHam>> regset;
    std::shared_ptr<abinitio::DataSet<DegHam>> degset;
    std::tie(regset, degset) = read_data(data);
    std::cout << "There are " << regset->size_int() << " data points in adiabatic representation\n"
              << "          " << degset->size_int() << " data points in composite representation\n\n";

    std::vector<std::shared_ptr<Energy>> energy_examples;
    auto energy_set = std::make_shared<abinitio::DataSet<Energy>>(energy_examples);
    if (args.gotArgument("energy_data")) {
        energy_set = read_energy(args.retrieve<std::vector<std::string>>("energy_data"));
        std::cout << "There are " << energy_set->size_int() << " data points without gradient\n\n";
    }

    double zero_point = 0.0;
    if (args.gotArgument("zero_point")) zero_point = args.retrieve<double>("zero_point");
    for (const auto & example : regset->examples()) example->subtract_ZeroPoint(zero_point);
    for (const auto & example : degset->examples()) example->subtract_ZeroPoint(zero_point);
    for (const auto & example : energy_set->examples()) example->subtract_ZeroPoint(zero_point);

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

    std::vector<std::pair<double, double>> energy_weight(Hdnet->NStates(), {0.0, 1.0});
    if (args.gotArgument("energy_weight")) {
        auto temp = args.retrieve<std::vector<double>>("energy_weight");
        if (temp.size() < 2 * Hdnet->NStates()) throw std::invalid_argument(
        "argument energy_weight: insufficient number of energy (reference, threshold) for each state");
        size_t count = 0;
        for (auto & ref_thresh : energy_weight) {
            ref_thresh.first  = temp[count    ];
            ref_thresh.second = temp[count + 1];
            count += 2;
        }
    }
    double dH_weight = unit * energy_weight[0].second;
    if (args.gotArgument("gradient_weight")) {
        dH_weight = args.retrieve<double>("gradient_weight");
        double sum_ethresh = 0.0;
        for (const auto & e_ref_thresh : energy_weight) sum_ethresh += e_ref_thresh.second;
        unit = dH_weight / (sum_ethresh / Hdnet->NStates());
        unit_square = unit * unit;
        std::cout << "According to user defined energy threshold and gradient threshold,\n"
                     "set gradient / energy scaling to " << unit << "\n\n";
    }
    for (const auto & example : regset->examples()) example->adjust_weight(energy_weight, dH_weight);
    // never alter the weight of degenerate examples
    for (const auto & example : energy_set->examples()) example->adjust_weight(energy_weight);

    bool regularized = args.gotArgument("regularization");
    if (regularized) {
        std::cout << "Got regularization strength, enable regularization\n\n";
        size_t NPars = 0;
        for (const at::Tensor & p : Hdnet->elements->parameters()) NPars += p.numel();
        // get regularization strength
        regularization = Hdnet->elements->parameters()[0].new_empty(NPars);
        std::string reg_prefix = args.retrieve<std::string>("regularization");
        std::ifstream ifs; ifs.open(reg_prefix + "_1-1_1.txt");
        if (ifs.good()) read_parameters(reg_prefix, regularization);
        else regularization.fill_(std::stod(reg_prefix));
        ifs.close();
        regularization.sqrt_(); // regularized residue = at::cat(residue, sqrt(stength) * weights)
        // get prior
        prior = Hdnet->elements->parameters()[0].new_zeros(NPars);
        if (args.gotArgument("prior")) {
            std::string prior_prefix = args.retrieve<std::string>("prior");
            std::ifstream ifs; ifs.open(prior_prefix + "_1-1_1.txt");
            if (ifs.good()) read_parameters(prior_prefix, prior);
            else prior.fill_(std::stod(prior_prefix));
        }
    }

    train::initialize();
    size_t max_iteration = 20;
    if (args.gotArgument("max_iteration")) max_iteration = args.retrieve<size_t>("max_iteration");
    train::trust_region::initialize(regset, degset, energy_set);
    train::trust_region::optimize(regularized, max_iteration);

    torch::save(Hdnet->elements, "Hd.net");

    std::cout << '\n';
    CL::utility::show_time(std::cout);
    std::cout << "Mission success\n";
}