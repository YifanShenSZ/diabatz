#include <CppLibrary/argparse.hpp>
#include <CppLibrary/chemistry.hpp>

#include <tchem/utility.hpp>
#include <tchem/linalg.hpp>

#include <Hd/kernel.hpp>

argparse::ArgumentParser parse_args(const size_t & argc, const char ** & argv) {
    CL::utility::echo_command(argc, argv, std::cout);
    std::cout << '\n';
    argparse::ArgumentParser parser("g-h plane double cone analysis based on diabatz");

    // required arguments
    parser.add_argument("-d","--diabatz", '+', false, "diabatz definition files");

    // optional arguments
    parser.add_argument("-g","--geometry", 1, true, "geometry to analyze double cone with (default = mex.xyz)");
    parser.add_argument("-t","--target",   1, true, "target and +1 state who cross (default = 0)");
    parser.add_argument("--g_vector", 1, true, "g_vector file (default = g.cart)");
    parser.add_argument("--h_vector", 1, true, "h_vector file (default = h.cart)");

    parser.parse_args(argc, argv);
    return parser;
}

int main(size_t argc, const char ** argv) {
    std::cout << "g-h plane double cone analysis based on diabatz\n"
              << "Yifan Shen 2021\n\n";
    argparse::ArgumentParser args = parse_args(argc, argv);
    CL::utility::show_time(std::cout);
    std::cout << '\n';

    std::vector<std::string> diabatz_inputs = args.retrieve<std::vector<std::string>>("diabatz");
    Hd::kernel Hdkernel(diabatz_inputs);

    std::string mex_file = "mex.xyz";
    if (args.gotArgument("geometry")) mex_file = args.retrieve<std::string>("geometry");
    CL::chem::xyz<double> mex(mex_file, true);
    std::vector<double> mex_coords = mex.coords();
    at::Tensor r_mex = at::from_blob(mex_coords.data(), mex_coords.size(), at::TensorOptions().dtype(torch::kFloat64));
    
    size_t target = 0;
    if (args.gotArgument("target")) target = args.retrieve<size_t>("target");

    std::string g_file = "g.cart", h_file = "h.cart";
    if (args.gotArgument("g_vector")) g_file = args.retrieve<std::string>("g_vector");
    if (args.gotArgument("h_vector")) h_file = args.retrieve<std::string>("h_vector");
    at::Tensor g = tchem::utility::read_vector(g_file); g /= g.norm();
    at::Tensor h = tchem::utility::read_vector(h_file); h /= h.norm();

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
        lower.push_back(energy[target    ].item<double>());
        upper.push_back(energy[target + 1].item<double>());
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