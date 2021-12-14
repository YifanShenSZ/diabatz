#include <CppLibrary/argparse.hpp>
#include <CppLibrary/utility.hpp>
#include <CppLibrary/chemistry.hpp>

#include "../include/global.hpp"

at::Tensor search_minimum(const at::Tensor & init_guess);

at::Tensor search_saddle(const at::Tensor & init_guess);

at::Tensor search_mex(const at::Tensor & _init_guess);

argparse::ArgumentParser parse_args(const size_t & argc, const char ** & argv) {
    CL::utility::echo_command(argc, argv, std::cout);
    std::cout << '\n';
    argparse::ArgumentParser parser("Critics: a critical geometry searcher");

    // required arguments
    parser.add_argument("-j","--job",        1, false, "job type: min, mex, sad");
    parser.add_argument("-f","--format",     1, false, "internal coordinate definition format (Columbus7, default)");
    parser.add_argument("-i","--IC",         1, false, "internal coordinate definition file");
    parser.add_argument("-t","--target",     1, false, "the target electronic state to search for critical geometry");
    parser.add_argument("-d","--diabatz",  '+', false, "diabatz definition files");
    parser.add_argument("-x","--xyz",        1, false, "initial guess xyz geometry file");

    // optional arguments
    parser.add_argument("-c","--fixed_coords", '+', true, "fix these internal coordinates during searching");
    parser.add_argument("-o","--output",         1, true, "output file, default = job.xyz");

    parser.parse_args(argc, argv);
    return parser;
}

int main(size_t argc, const char ** argv) {
    std::cout << "Critics: a critical geometry searcher\n"
              << "Yifan Shen 2021\n\n";
    argparse::ArgumentParser args = parse_args(argc, argv);
    CL::utility::show_time(std::cout);
    std::cout << '\n';

    std::string job = args.retrieve<std::string>("job");
    std::cout << "Job type: " + job << '\n';

    std::string format = args.retrieve<std::string>("format");
    std::string IC     = args.retrieve<std::string>("IC");
    intcoordset = std::make_shared<tchem::IC::IntCoordSet>(format, IC);

    target_state = args.retrieve<size_t>("target");
    std::cout << "The target electronic state is " << target_state << '\n';
    target_state -= 1;

    std::vector<std::string> diabatz_inputs = args.retrieve<std::vector<std::string>>("diabatz");
    Hdkernel = std::make_shared<Hd::kernel>(diabatz_inputs);

    if (target_state >= Hdkernel->NStates()) {
        std::cerr << "Error: There are only " << Hdkernel->NStates() << " electronic states, "
                  << "so the target state should not exceed that\n";
        throw std::invalid_argument("Bad target state");
    }

    std::string guess_file = args.retrieve<std::string>("xyz");
    CL::chem::xyz<double> init_geom(guess_file, true);
    std::vector<double> init_coords = init_geom.coords();
    at::Tensor init_r = at::from_blob(init_coords.data(), init_coords.size(), at::TensorOptions().dtype(torch::kFloat64));
    at::Tensor init_q = (*intcoordset)(init_r);

    std::vector<size_t> fixed_coords;
    if (args.gotArgument("fixed_coords")) {
        fixed_coords = args.retrieve<std::vector<size_t>>("fixed_coords");
        for (size_t & fixed_coord : fixed_coords) fixed_coord -= 1;
    }
    fixed_intcoord = std::make_shared<Fixed_intcoord>(intcoordset->size(), fixed_coords, init_q);

    at::Tensor final_r;
    if      (job == "min") final_r = search_minimum(init_r);
    else if (job == "sad") final_r = search_saddle (init_r);
    else if (job == "mex") final_r = search_mex    (init_r);
    else throw std::invalid_argument("Unsupported job type");

    size_t cartdim = init_coords.size();
    std::vector<double> final_coords(cartdim);
    std::memcpy(final_coords.data(), final_r.data_ptr<double>(), cartdim * sizeof(double));
    CL::chem::xyz<double> final_geom(init_geom.symbols(), final_coords, true);
    std::string output = job + ".xyz";
    if (args.gotArgument("output")) output = args.retrieve<std::string>("output");
    final_geom.print(output);

    std::cout << '\n';
    CL::utility::show_time(std::cout);
    std::cout << "Mission success\n";
}