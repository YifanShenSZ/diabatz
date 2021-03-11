#include <CppLibrary/argparse.hpp>
#include <CppLibrary/chemistry.hpp>

#include <tchem/utility.hpp>
#include <tchem/linalg.hpp>

#include <Hd/kernel.hpp>

argparse::ArgumentParser parse_args(const size_t & argc, const char ** & argv) {
    CL::utility::echo_command(argc, argv, std::cout);
    std::cout << '\n';
    argparse::ArgumentParser parser("g-h plane double cone for diabatz version 0");

    // required arguments
    parser.add_argument("-f","--format",         1, false, "internal coordinate definition format (Columbus7, default)");
    parser.add_argument("-i","--IC",             1, false, "internal coordinate definition file");
    parser.add_argument("-s","--SAS",            1, false, "symmetry adaptation and scale definition file");
    parser.add_argument("-n","--net",            1, false, "diabatic Hamiltonian network definition file");
    parser.add_argument("-c","--checkpoint",     1, false, "a trained Hd parameter to continue with");
    parser.add_argument("-l","--input_layers", '+', false, "network input layer definition files");

    parser.parse_args(argc, argv);
    return parser;
}

int main(size_t argc, const char ** argv) {
    std::cout << "g-h plane double cone for diabatz version 0\n"
              << "Yifan Shen 2021\n\n";
    argparse::ArgumentParser args = parse_args(argc, argv);
    CL::utility::show_time(std::cout);
    std::cout << '\n';

    std::string format = args.retrieve<std::string>("format");
    std::string IC     = args.retrieve<std::string>("IC");
    std::string SAS    = args.retrieve<std::string>("SAS");
    std::string net_in = args.retrieve<std::string>("net");
    std::string chk    = args.retrieve<std::string>("checkpoint");
    std::vector<std::string> input_layers = args.retrieve<std::vector<std::string>>("input_layers");
    Hd::kernel Hdkernel(format, IC, SAS, net_in, chk, input_layers);

    CL::chem::xyz<double> mex("mex.xyz", true);
    std::vector<double> mex_coords = mex.coords();
    at::Tensor r_mex = at::from_blob(mex_coords.data(), mex_coords.size(), at::TensorOptions().dtype(torch::kFloat64));
    
    at::Tensor g = tchem::utility::read_vector("g.cart"); g /= g.norm();
    at::Tensor h = tchem::utility::read_vector("h.cart"); h /= h.norm();

    int64_t Ng = 10, Nh = 10;
    double dg = 0.01, dh = 0.01;
    std::vector<double> gmesh, hmesh, lower, upper;
    for (int64_t i = -Ng; i <= Ng; i++)
    for (int64_t j = -Nh; j <= Nh; j++) {
        at::Tensor r = r_mex + i * dg * g + j * dh * h;
        at::Tensor Hd, dHd;
        std::tie(Hd, dHd) = Hdkernel.compute_Hd_dHd(r);
        at::Tensor energy, states;
        std::tie(energy, states) = Hd.symeig();
        gmesh.push_back(i * dg);
        hmesh.push_back(j * dh);
        lower.push_back(energy[0].item<double>());
        upper.push_back(energy[1].item<double>());
    }

    std::ofstream ofs;
    ofs.open("g-mesh.txt");
    for (const double & g : gmesh) ofs << g << '\n';
    ofs.close();
    ofs.open("h-mesh.txt");
    for (const double & h : hmesh) ofs << h << '\n';
    ofs.close();
    ofs.open("lower.txt");
    for (const double & e : lower) ofs << e << '\n';
    ofs.close();
    ofs.open("upper.txt");
    for (const double & e : upper) ofs << e << '\n';
    ofs.close();

    std::cout << '\n';
    CL::utility::show_time(std::cout);
    std::cout << "Mission success\n";
}