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
    parser.add_argument("-d","--data",         '+', false, "data set list files or directories");

    // optional arguments
    parser.add_argument("--energy_data",      '+', true, "data set list files or directories without gradient");
    parser.add_argument("-z","--zero_point",    1, true, "zero of potential energy, default = 0");
    parser.add_argument("--energy_weight"  ,  '+', true, "energy (reference, threshold) for each state in weight adjustment, default = (0, 1)");
    parser.add_argument("--gradient_weight",    1, true, "gradient threshold in weight adjustment, default = infer from energy threshold");
    parser.add_argument("-g","--guess_diag",  '+', true, "initial guess of Hd diagonal, default = pytorch initialization");
    parser.add_argument("-c","--checkpoint",    1, true, "a trained Hd parameter to continue from");

    // regularization arguments
    parser.add_argument("-r","--regularization", 1, true, "enable regularization and set strength, can be a scalar or files regularization_state1-state2_layer.txt");
    parser.add_argument("-p","--prior",          1, true, "prior for regularization are taken from files prior_state1-state2_layer.txt");

    // optimizer arguments
    parser.add_argument("-o","--optimizer",     1, true, "trust_region, Adam, SGD (default = trust_region)");
    parser.add_argument("-m","--max_iteration", 1, true, "default = 100");
    parser.add_argument("--batch_size",         1, true, "for Adam or SGD, default = 32");
    parser.add_argument("--learning_rate",      1, true, "for Adam or SGD, default = 1e-3");
    parser.add_argument("--opt_chk",            1, true, "optimizer checkpoint for Adam or SGD");

    parser.parse_args(argc, argv);
    return parser;
}

void read_parameters(const std::string & prefix, at::Tensor & x) {
    auto pmat = Hdnet->parameters();
    size_t start = 0;
    for (size_t istate = 0     ; istate < Hdnet->NStates(); istate++)
    for (size_t jstate = istate; jstate < Hdnet->NStates(); jstate++) {
        std::string prefix_now = prefix + "_" + std::to_string(istate + 1) + "-" + std::to_string(jstate + 1) + "_";
        const auto & ps = pmat[istate][jstate];
        // The 1st layer is interpretable
        std::string file = prefix_now + "1.txt";
        std::ifstream ifs; ifs.open(file);
        at::Tensor A = ps[0].new_empty(ps[0].sizes());
        for (size_t i = 0; i < A.size(1); i++) {
            std::string line;
            std::getline(ifs, line);
            if (! ifs.good()) throw CL::utility::file_error(file);
            std::getline(ifs, line);
            if (! ifs.good()) throw CL::utility::file_error(file);
            auto strs = CL::utility::split(line);
            if (strs.size() != A.size(0)) throw std::invalid_argument("inconsisten line");
            for (size_t j = 0; j < A.size(0); j++) A[j][i].fill_(std::stod(strs[j]));
        }
        size_t stop = start + A.numel();
        x.slice(0, start, stop).copy_(A.view(A.numel()));
        start = stop;
        if (Hdnet->irreds()[istate][jstate] == 0) {
            at::Tensor b = ps[1].new_empty(ps[1].sizes());
            std::string line;
            std::getline(ifs, line);
            if (! ifs.good()) throw CL::utility::file_error(file);
            std::getline(ifs, line);
            if (! ifs.good()) throw CL::utility::file_error(file);
            auto strs = CL::utility::split(line);
            if (strs.size() != b.size(0)) throw std::invalid_argument("inconsisten line");
            for (size_t j = 0; j < A.size(0); j++) b[j].fill_(std::stod(strs[j]));
            size_t stop = start + b.numel();
            x.slice(0, start, stop).copy_(b);
            start = stop;
        }
        ifs.close();
        // The other layers are uninterpretable
        if (Hdnet->irreds()[istate][jstate] == 0)
        for (size_t layer = 2; layer < ps.size(); layer += 2) {
            std::string file = prefix_now + std::to_string(layer / 2 + 1) + ".txt";
            std::ifstream ifs; ifs.open(file);
            at::Tensor A = ps[layer].new_empty(ps[layer].sizes());
            std::string line;
            std::getline(ifs, line);
            for (size_t i = 0; i < A.size(1); i++)
            for (size_t j = 0; j < A.size(0); j++) {
                double dbletemp;
                ifs >> dbletemp;
                if (! ifs.good()) throw CL::utility::file_error(file);
                A[j][i].fill_(dbletemp);
            }
            at::Tensor b = ps[layer + 1].new_empty(ps[layer + 1].sizes());
            ifs >> line;
            for (size_t i = 0; i < b.size(0); i++) {
                double dbletemp;
                ifs >> dbletemp;
                if (! ifs.good()) throw CL::utility::file_error(file);
                b[i].fill_(dbletemp);
            }
            size_t stop = start + A.numel();
            x.slice(0, start, stop).copy_(A.view(A.numel()));
            start = stop;
            stop = start + b.numel();
            x.slice(0, start, stop).copy_(b);
            start = stop;
            ifs.close();
        }
        else
        for (size_t layer = 1; layer < ps.size(); layer++) {
            std::string file = prefix_now + std::to_string(layer + 1) + ".txt";
            std::ifstream ifs; ifs.open(file);
            at::Tensor A = ps[layer].new_empty(ps[layer].sizes());
            std::string line;
            std::getline(ifs, line);
            for (size_t i = 0; i < A.size(1); i++)
            for (size_t j = 0; j < A.size(0); j++) {
                double dbletemp;
                ifs >> dbletemp;
                if (! ifs.good()) throw CL::utility::file_error(file);
                A[j][i].fill_(dbletemp);
            }
            size_t stop = start + A.numel();
            x.slice(0, start, stop).copy_(A.view(A.numel()));
            start = stop;
            ifs.close();
        }
    }
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
    sasicset = std::make_shared<SASIC::SASICSet>(format, IC, SAS);

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
              << "          " << degset->size_int() << " data points in composite representation\n";

    std::shared_ptr<abinitio::DataSet<Energy>> energy_set;
    if (args.gotArgument("energy_data")) {
        energy_set = read_energy(args.retrieve<std::vector<std::string>>("energy_data"));
        std::cout << "          " << energy_set->size_int() << " data points without gradient\n";
    }
    std::cout << '\n';

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
    // never alter the weight of no-gradient examples

    bool regularized = args.gotArgument("regularization");
    if (regularized) {
        std::cout << "Got regularization strength, enable regularization\n\n";
        size_t NPars = 0;
        for (const at::Tensor & p : Hdnet->elements->parameters()) NPars += p.numel();
        // Get regularization strength
        regularization = Hdnet->elements->parameters()[0].new_empty(NPars);
        std::string reg_prefix = args.retrieve<std::string>("regularization");
        std::ifstream ifs; ifs.open(reg_prefix + "_1-1_1.txt");
        if (ifs.good()) read_parameters(reg_prefix, regularization);
        else regularization.fill_(std::stod(reg_prefix));
        ifs.close();
        // Get prior
        prior = Hdnet->elements->parameters()[0].new_empty(NPars);
        if (! args.gotArgument("prior")) throw std::invalid_argument("Prior must be provided for regularization");
        std::string prior_prefix = args.retrieve<std::string>("prior");
        read_parameters(prior_prefix, prior);
    }

    train::initialize();
    size_t max_iteration = 100;
    if (args.gotArgument("max_iteration")) max_iteration = args.retrieve<size_t>("max_iteration");
    std::string optimizer = "trust_region";
    if (args.gotArgument("optimizer")) optimizer = args.retrieve<std::string>("optimizer");
    if (optimizer == "trust_region") {
        std::cout << "Optimizer is trust region\n\n";
        train::trust_region::initialize(regset, degset, energy_set);
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
            train::torch_optim::Adam(regset, degset, energy_set, max_iteration, batch_size, learning_rate, opt_chk);
        }
        else if (optimizer == "SGD") {
            std::cout << "Optimizer is stochastic gradient descent (SGD)\n\n";
            train::torch_optim::SGD(regset, degset, energy_set, max_iteration, batch_size, learning_rate, opt_chk);
        }
        else throw std::invalid_argument("Unsupported optimizer " + optimizer);
    }

    std::cout << '\n';
    CL::utility::show_time(std::cout);
    std::cout << "Mission success\n";
}