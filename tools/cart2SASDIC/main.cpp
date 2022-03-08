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
    parser.add_argument("-x","--xyz",      1, false, "input xyz geometry (output to `prefix`.int)");

    // optional argument
    parser.add_argument("-j","--Jacobian", 0, true, "also compute Jacobian (output to `prefix`.log)");

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

    std::string input_file = CL::utility::split(geom_xyz, '/').back();
    std::string prefix = CL::utility::split(input_file, '.')[0];

    if (! args.gotArgument("Jacobian")) {
        // Cartesian coordinate -> internal coordinate
        at::Tensor q = sasicset.IntCoordSet::operator()(r);
        // internal coordinate -> CNPI group symmetry adapted internal coordinate
        std::vector<at::Tensor> qs = sasicset(q);
        // output
        std::ofstream ofs; ofs.open(prefix + ".int");
        for (size_t i = 0; i < qs.size(); i++) {
            ofs << "Irreducible " << i + 1 << ":\n";
            const double * p = qs[i].data_ptr<double>();
            for (size_t j = 0; j < qs[i].numel(); j++) ofs << std::fixed << std::setw(18) << std::setprecision(15) << p[j] << '\n';
            ofs << '\n';
        }
        ofs.close();
    }
    else {
        // Cartesian coordinate -> internal coordinate
        at::Tensor q, J;
        std::tie(q, J) = sasicset.compute_IC_J(r);
        q.requires_grad_(true);
        // internal coordinate -> CNPI group symmetry adapted internal coordinate
        std::vector<at::Tensor> qs = sasicset(q);
        std::vector<at::Tensor> Js = std::vector<at::Tensor>(qs.size());
        for (size_t i = 0; i < qs.size(); i++) {
            Js[i] = qs[i].new_empty({qs[i].size(0), q.size(0)});
            for (size_t j = 0; j < qs[i].size(0); j++) {
                std::vector<at::Tensor> g = torch::autograd::grad({qs[i][j]}, {q}, {}, true);
                Js[i][j].copy_(g[0]);
            }
            Js[i] = Js[i].mm(J);
        }
        J = at::cat(Js);
        // stop autograd tracking
        for (at::Tensor & q : qs) q.detach_();
        // output
        std::ofstream ofs; ofs.open(prefix + ".int");
        for (size_t i = 0; i < qs.size(); i++) {
            ofs << "Irreducible " << i + 1 << ":\n";
            const double * p = qs[i].data_ptr<double>();
            for (size_t j = 0; j < qs[i].numel(); j++) ofs << std::fixed << std::setw(18) << std::setprecision(15) << p[j] << '\n';
            ofs << '\n';
        }
        ofs.close();
        size_t  intdim = sasicset.intdim(),
               cartdim = r.size(0);
        std::vector<double> freqs(intdim, 0.0); // irred i coord j has "freq" ij
        size_t count = 0;
        for (size_t i = 0; i < qs.size(); i++)
        for (size_t j = 0; j < qs[i].numel(); j++) {
            freqs[count] = ((i + 1) * 100 + j + 1) * 4.556335830019422e-6;
            count++;
        }
        CL::utility::matrix<double> modes(intdim, cartdim);
        for (size_t i = 0; i <  intdim; i++)
        for (size_t j = 0; j < cartdim; j++)
        modes[i][j] = J[i][j].item<double>();
        CL::chem::xyz_vib<double> geom_J(geom, freqs, modes);
        geom_J.print(prefix + ".log");
    }

    CL::utility::show_time(std::cout);
    std::cout << "Mission success\n";
}