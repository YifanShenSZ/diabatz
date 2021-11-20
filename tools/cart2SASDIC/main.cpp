#include <CppLibrary/argparse.hpp>
#include <CppLibrary/utility.hpp>
#include <CppLibrary/chemistry.hpp>

#include <SASDIC/SASDICSet.hpp>

argparse::ArgumentParser parse_args(const size_t & argc, const char ** & argv) {
    CL::utility::echo_command(argc, argv, std::cout);
    std::cout << '\n';
    argparse::ArgumentParser parser("cart2SASDIC: Convert geometry from Cartesian coordinate to symmetry adapted and scaled dimensionless internal coordinate");

    // required arguments
    parser.add_argument("-f","--format",   1, false, "internal coordinate definition format (Columbus7, default)");
    parser.add_argument("-i","--IC",       1, false, "internal coordinate definition file");
    parser.add_argument("-s","--SAS",      1, false, "symmetry adaptation and scale definition file");
    parser.add_argument("-x","--xyz",      1, false, "input xyz geometry");

    // optional argument
    parser.add_argument("-o","--output", 1, true, "output symmetry adapted and scaled internal coordinate (default = `input`.int)");

    parser.parse_args(argc, argv);
    return parser;
}

int main(size_t argc, const char ** argv) {
    std::cout << "Convert geometry from Cartesian coordinate to symmetry adapted and scaled dimensionless internal coordinate\n";
    argparse::ArgumentParser args = parse_args(argc, argv);
    CL::utility::show_time(std::cout);
    std::cout << '\n';

    std::string format = args.retrieve<std::string>("format");
    std::string IC     = args.retrieve<std::string>("IC");
    std::string SAS    = args.retrieve<std::string>("SAS");
    SASDIC::SASDICSet sasicset(format, IC, SAS);

    std::string geom_xyz = args.retrieve<std::string>("xyz");
    CL::chem::xyz<double> geom(geom_xyz, true);
    std::vector<double> coords = geom.coords();
    at::Tensor r = at::from_blob(coords.data(), coords.size(), at::TensorOptions().dtype(torch::kFloat64));
    at::Tensor q = sasicset.IntCoordSet::operator()(r);
    std::vector<at::Tensor> SASgeom = sasicset(q);

    std::ofstream ofs;
    if (args.gotArgument("output")) ofs.open(args.retrieve<std::string>("output"));
    else {
        std::string input_file = CL::utility::split(geom_xyz, '/').back();
        std::string prefix = CL::utility::split(input_file, '.')[0];
        ofs.open(prefix + ".int");
    }
    for (size_t i = 0; i < SASgeom.size(); i++) {
        ofs << "Irreducible " << i + 1 << ":\n";
        const double * p = SASgeom[i].data_ptr<double>();
        for (size_t j = 0; j < SASgeom[i].numel(); j++) ofs << std::fixed << std::setw(18) << std::setprecision(15) << p[j] << '\n';
        ofs << '\n';
    }
    ofs.close();

    CL::utility::show_time(std::cout);
    std::cout << "Mission success\n";
}