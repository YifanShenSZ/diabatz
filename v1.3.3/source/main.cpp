#include <CppLibrary/argparse.hpp>
#include <CppLibrary/utility.hpp>

#include "../include/global.hpp"
#include "../include/data.hpp"
#include "../include/train.hpp"
#include "../include/utility.hpp"

argparse::ArgumentParser parse_args(const size_t & argc, const char ** & argv) {
    CL::utility::echo_command(argc, argv, std::cout);
    std::cout << '\n';
    argparse::ArgumentParser parser("Diabatz version 1.3.3");

    // required arguments
    parser.add_argument("-f","--format", 1, false, "internal coordinate definition format (Columbus7, default)");
    parser.add_argument("-i","--IC",     1, false, "internal coordinate definition file");
    parser.add_argument("-s","--SAS",    1, false, "symmetry adaptation and scale definition file");
    parser.add_argument("-d","--data", '+', false, "data set list files or directories");
    // network 1
    parser.add_argument("--net1",            1, false, "diabatic Hamiltonian network definition file");
    parser.add_argument("--input_layers1", '+', false, "network input layer definition files");
    // network 2
    parser.add_argument("--net2",            1, false, "diabatic Hamiltonian network definition file");
    parser.add_argument("--input_layers2", '+', false, "network input layer definition files");

    // optional arguments
    parser.add_argument("--energy_data",      '+', true, "data set list files or directories without gradient");
    parser.add_argument("-z","--zero_point",    1, true, "zero of potential energy, default = 0");
    parser.add_argument("--energy_weight"  ,  '+', true, "energy (reference, threshold) for each state in weight adjustment, default = (0, 1)");
    parser.add_argument("--gradient_weight",    1, true, "gradient threshold in weight adjustment, default = infer from energy threshold");
    // network 1
    parser.add_argument("--checkpoint1",     1, true, "a trained Hd parameter to continue from");
    parser.add_argument("--regularization1", 1, true, "regularization strength, can be a scalar or files regularization_state1-state2_layer.txt");
    parser.add_argument("--prior1",          1, true, "prior for regularization, can be a scalar or files prior_state1-state2_layer.txt, default = 0");
    // network 2
    parser.add_argument("--checkpoint2", 1, true, "a trained Hd parameter to continue from");
    parser.add_argument("--regularization2", 1, true, "regularization strength, can be a scalar or files regularization_state1-state2_layer.txt");
    parser.add_argument("--prior2",          1, true, "prior for regularization, can be a scalar or files prior_state1-state2_layer.txt, default = 0");

    // optimizer arguments
    parser.add_argument("-m","--max_iteration", 1, true, "default = 100");

    parser.parse_args(argc, argv);
    return parser;
}

int main(size_t argc, const char ** argv) {
    std::cout << "Diabatz version 1.3.3\n"
              << "Yifan Shen 2022\n\n";
    argparse::ArgumentParser args = parse_args(argc, argv);
    CL::utility::show_time(std::cout);
    std::cout << '\n';

    std::string format = args.retrieve<std::string>("format"),
                IC     = args.retrieve<std::string>("IC"),
                SAS    = args.retrieve<std::string>("SAS");
    sasicset = std::make_shared<SASDIC::SASDICSet>(format, IC, SAS);

    Hdnet1 = std::make_shared<obnet::symat>(args.retrieve<std::string>("net1"));
    Hdnet1->train();
    if (args.gotArgument("checkpoint1")) torch::load(Hdnet1->elements, args.retrieve<std::string>("checkpoint1"));

    Hdnet2 = std::make_shared<obnet::symat>(args.retrieve<std::string>("net2"));
    Hdnet2->train();
    if (args.gotArgument("checkpoint2")) torch::load(Hdnet2->elements, args.retrieve<std::string>("checkpoint2"));

    std::vector<std::string> input_layers1 = args.retrieve<std::vector<std::string>>("input_layers1");
    if (input_layers1.size() != (Hdnet1->NStates() + 1) * Hdnet1->NStates() / 2) throw std::invalid_argument(
    "The number of input layers must match the number of Hd upper-triangle elements");
    input_generator1 = std::make_shared<InputGenerator>(Hdnet1->NStates(), Hdnet1->irreds(), input_layers1, sasicset->NSASDICs());

    std::vector<std::string> input_layers2 = args.retrieve<std::vector<std::string>>("input_layers2");
    if (input_layers2.size() != (Hdnet2->NStates() + 1) * Hdnet2->NStates() / 2) throw std::invalid_argument(
    "The number of input layers must match the number of Hd upper-triangle elements");
    input_generator2 = std::make_shared<InputGenerator>(Hdnet2->NStates(), Hdnet2->irreds(), input_layers2, sasicset->NSASDICs());

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

    std::vector<std::pair<double, double>> energy_weight(Hdnet1->NStates(), {0.0, 1.0});
    if (args.gotArgument("energy_weight")) {
        auto temp = args.retrieve<std::vector<double>>("energy_weight");
        if (temp.size() < 2 * Hdnet1->NStates()) throw std::invalid_argument(
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
        unit = dH_weight / (sum_ethresh / Hdnet1->NStates());
        unit_square = unit * unit;
        std::cout << "According to user defined energy threshold and gradient threshold,\n"
                     "set gradient / energy scaling to " << unit << "\n\n";
    }
    for (const auto & example : regset->examples()) example->adjust_weight(energy_weight, dH_weight);
    // never alter the weight of degenerate examples
    for (const auto & example : energy_set->examples()) example->adjust_weight(energy_weight);

    int64_t NPars1 = 0, NPars2 = 0;
    for (const at::Tensor& p : Hdnet1->elements->parameters()) NPars1 += p.numel();
    for (const at::Tensor& p : Hdnet2->elements->parameters()) NPars2 += p.numel();
    regularization = Hdnet1->elements->parameters()[0].new_zeros(NPars1 + NPars2);
    prior = Hdnet1->elements->parameters()[0].new_zeros(NPars1 + NPars2);
    // read regularization
    if (args.gotArgument("regularization1")) {
        at::Tensor regularization1 = regularization.slice(0, 0, NPars1);
        std::string reg_prefix = args.retrieve<std::string>("regularization1");
        std::ifstream ifs; ifs.open(reg_prefix + "_1-1_1.txt");
        if (ifs.good()) read_parameters(Hdnet1, reg_prefix, regularization1);
        else regularization.fill_(std::stod(reg_prefix));
        ifs.close();
    }
    if (args.gotArgument("regularization2")) {
        at::Tensor regularization2 = regularization.slice(0, NPars1, NPars1 + NPars2);
        std::string reg_prefix = args.retrieve<std::string>("regularization2");
        std::ifstream ifs; ifs.open(reg_prefix + "_1-1_1.txt");
        if (ifs.good()) read_parameters(Hdnet2, reg_prefix, regularization2);
        else regularization.fill_(std::stod(reg_prefix));
        ifs.close();
    }
    regularization.sqrt_(); // regularized residue = at::cat(residue, sqrt(stength) * weights)
    // read prior
    if (args.gotArgument("prior1")) {
        at::Tensor prior1 = prior.slice(0, 0, NPars1);
        std::string prior_prefix = args.retrieve<std::string>("prior1");
        std::ifstream ifs; ifs.open(prior_prefix + "_1-1_1.txt");
        if (ifs.good()) read_parameters(Hdnet1, prior_prefix, prior1);
        else prior.fill_(std::stod(prior_prefix));
    }
    if (args.gotArgument("prior2")) {
        at::Tensor prior2 = prior.slice(0, NPars1, NPars1 + NPars2);
        std::string prior_prefix = args.retrieve<std::string>("prior2");
        std::ifstream ifs; ifs.open(prior_prefix + "_1-1_1.txt");
        if (ifs.good()) read_parameters(Hdnet1, prior_prefix, prior2);
        else prior.fill_(std::stod(prior_prefix));
    }

    // define feature scaling by the regular data set
    CL::utility::matrix<at::Tensor> shift1, width1, shift2, width2;
    std::tie(shift1, width1, shift2, width2) = statisticize_regset(regset);
    for (const auto & example : regset->examples()) example->scale_features(shift1, width1, shift2, width2);
    for (const auto & example : degset->examples()) example->scale_features(shift1, width1, shift2, width2);
    for (const auto & example : energy_set->examples()) example->scale_features(shift1, width1, shift2, width2);
    // if current parameters come from a checkpoint,
    // rescale Hdnet parameters according to feature scaling
    // so that Hdnet still outputs a same value for a same geometry;
    // else Xavier initialization is good
    if (args.gotArgument("checkpoint1")) rescale_Hdnet(Hdnet1, shift1, width1);
    if (args.gotArgument("checkpoint2")) rescale_Hdnet(Hdnet2, shift2, width2);
    // if enabled regularization, rescale prior
    if (args.gotArgument("prior1")) {
        at::Tensor prior1 = prior.slice(0, 0, NPars1);
        rescale_parameters(Hdnet1, shift1, width1, prior1);
    }
    if (args.gotArgument("prior2")) {
        at::Tensor prior2 = prior.slice(0, NPars1, NPars1 + NPars2);
        rescale_parameters(Hdnet2, shift2, width2, prior2);
    }

    train::initialize();
    size_t max_iteration = 100;
    if (args.gotArgument("max_iteration")) max_iteration = args.retrieve<size_t>("max_iteration");
    train::trust_region::initialize(regset, degset, energy_set);
    train::trust_region::optimize(max_iteration);

    unscale_Hdnet(Hdnet1, shift1, width1);
    unscale_Hdnet(Hdnet2, shift2, width2);
    torch::save(Hdnet1->elements, "Hd1.net");
    torch::save(Hdnet2->elements, "Hd2.net");

    std::cout << '\n';
    CL::utility::show_time(std::cout);
    std::cout << "Mission success\n";
}