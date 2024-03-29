#include <CppLibrary/argparse.hpp>
#include <CppLibrary/chemistry.hpp>

#include <tchem/utility.hpp>
#include <tchem/linalg.hpp>
#include <tchem/chemistry.hpp>

#include <abinitio/SAgeometry.hpp>

#include <Hd/Kernel.hpp>

#include "../include/CNPI.hpp"
#include "../include/routines.hpp"

argparse::ArgumentParser parse_args(const size_t & argc, const char ** & argv) {
    CL::utility::echo_command(argc, argv, std::cout);
    std::cout << '\n';
    argparse::ArgumentParser parser("Generate *vibronics* input");

    // diabatz definition
    parser.add_argument("-f","--format",         1, false, "internal coordinate definition format (Columbus7, default)");
    parser.add_argument("-i","--IC",             1, false, "internal coordinate definition file");
    parser.add_argument("-s","--SAS",            1, false, "symmetry adaptation and scale definition file");
    parser.add_argument("-n","--net",            1, false, "diabatic Hamiltonian network definition file");
    parser.add_argument("-c","--checkpoint",     1, false, "a trained Hd parameter to continue with");
    parser.add_argument("-l","--input_layers", '+', false, "network input layer definition files");

    // molecule definition
    parser.add_argument("-m","--mass",  1, false, "the masses of atoms");
    parser.add_argument("--init_geom",  1, false, "intial-state equilibrium geometry");
    parser.add_argument("--final_geom", 1, false, " final-state equilibrium geometry");

    // init-state Hessian
    parser.add_argument("--init_hess", 1, false, "intial-state Columbus Hessian");

    // analyzation
    parser.add_argument("--contour", 1, false, "contour value");

    // optional argument
    parser.add_argument("--init_CNPI2point", '+', true, "how CNPI group irreducibles map to point group irreducibles at initial-state equilibrium geometry");

    parser.parse_args(argc, argv);
    return parser;
}

int main(size_t argc, const char ** argv) {
    std::cout << "Generate *vibronics* input\n"
              << "Yifan Shen 2021\n\n";
    argparse::ArgumentParser args = parse_args(argc, argv);
    CL::utility::show_time(std::cout);
    std::cout << '\n';

    std::string format = args.retrieve<std::string>("format"),
                IC     = args.retrieve<std::string>("IC"),
                SAS    = args.retrieve<std::string>("SAS");
    sasicset = std::make_shared<SASDIC::SASDICSet>(format, IC, SAS);

    std::string net = args.retrieve<std::string>("net"),
                chk = args.retrieve<std::string>("checkpoint");
    std::vector<std::string> input_layers = args.retrieve<std::vector<std::string>>("input_layers");
    Hd::Kernel HdKernel(format, IC, SAS, net, chk, input_layers);

    std::string mass_file = args.retrieve<std::string>("mass"),
                final_geom_file = args.retrieve<std::string>("final_geom"),
                 init_geom_file = args.retrieve<std::string>( "init_geom");
    CL::chem::xyz_mass<double> final_geom(final_geom_file, mass_file, true),
                                init_geom( init_geom_file, mass_file, true);
    std::vector<double>  init_coords =  init_geom.coords(),
                        final_coords = final_geom.coords();
    at::Tensor  init_r = at::from_blob( init_coords.data(),  init_coords.size(), at::TensorOptions().dtype(torch::kFloat64)),
               final_r = at::from_blob(final_coords.data(), final_coords.size(), at::TensorOptions().dtype(torch::kFloat64));

    // initial-state equilibrium internal coordinate geometry
    std::vector<at::Tensor> init_CNPI_qs, init_CNPI_Js;
    std::tie(init_CNPI_qs, init_CNPI_Js) = cart2CNPI(init_r);
    std::vector<size_t> init_CNPI2point(init_CNPI_qs.size());
    if (args.gotArgument("init_CNPI2point")) {
        init_CNPI2point = args.retrieve<std::vector<size_t>>("init_CNPI2point");
        for (size_t & irred : init_CNPI2point) irred -= 1;
    }
    else {
        for (size_t i = 1; i < init_CNPI_qs.size(); i++)
        if (init_CNPI_qs[i].norm().item<double>() > 1e-6) {
            std::cerr << "Warning: the point group of the initial-state equilibirum geometry may not be isomorphic to the CNPI group\n"
                      << "Please consider adding argument --init_CPNI2point\n";
            break;
        }
        std::iota(init_CNPI2point.begin(), init_CNPI2point.end(), 0);
    }
    // create mapping from point group to CNPI group
    // point irreducible i contains CNPI irreducibles point2CNPI[i]
    size_t n_point_irreds = *std::max_element(init_CNPI2point.begin(), init_CNPI2point.end()) + 1;
    std::vector<std::vector<size_t>> init_point2CNPI(n_point_irreds);
    for (size_t i = 0; i < n_point_irreds; i++)
    for (size_t j = 0; j < init_CNPI2point.size(); j++)
    if (init_CNPI2point[j] == i) init_point2CNPI[i].push_back(j);
    std::vector<at::Tensor> init_qs = cat(init_CNPI_qs, init_point2CNPI),
                            init_Js = cat(init_CNPI_Js, init_point2CNPI);

    // final-state equilibrium internal coordinate geometry
    std::vector<at::Tensor> final_qs, final_Js;
    std::tie(final_qs, final_Js) = cart2CNPI(final_r);
    for (size_t i = 1; i < final_qs.size(); i++)
    if (final_qs[i].norm().item<double>() > 1e-6) throw std::invalid_argument(
    "the point group of the final-state equilibirum geometry must be isomorphic to the CNPI group\n");

    std::string init_hess_file = args.retrieve<std::string>("init_hess");
    at::Tensor init_carthess = read_Columbus(init_r, init_hess_file);
    std::vector<at::Tensor> init_Hs = Hessian_cart2int(init_r, init_CNPI2point, init_carthess);
    tchem::chem::SANormalMode init_vib(init_geom.masses(), init_Js, init_Hs);
    init_vib.kernel();

    at::Tensor final_intddHd = compute_intddHd(final_r, HdKernel);
    at::Tensor final_inthess;
    // Determine representation
    at::Tensor final_Hd, final_dHd;
    std::tie(final_Hd, final_dHd) = HdKernel.compute_Hd_dHd(final_r);
    at::Tensor final_energy, final_state;
    std::tie(final_energy, final_state) = final_Hd.symeig(true);
    if ((final_energy[1] - final_energy[0]).item<double>() < 1e-4) {
        std::cout << "The final-state equilibirum geometry is a denegerate point, using\n"
                  << "    Hessian = (▽▽Hd[0][0] + ▽▽Hd[1][1]) / 2\n\n";
        final_inthess = (final_intddHd[0][0] + final_intddHd[1][1]) / 2.0;
    }
    else {
        std::cout << "The final-state equilibirum geometry is a minimum, using\n"
                  << "    Hessian = ▽▽Hd[0][0]\n\n";
        final_inthess = final_intddHd[0][0];
    }
    // Split internal coordinate Hessian to different irreducibles
    std::vector<at::Tensor> final_Hs(final_qs.size());
    size_t start = 0;
    for (size_t i = 0; i < final_Hs.size(); i++) {
        size_t end = start + final_qs[i].size(0);
        final_Hs[i] = final_inthess.slice(0, start, end).slice(1, start, end);
        start = end;
    }
    // Generate normal mode
    tchem::chem::SANormalMode final_vib(final_geom.masses(), final_Js, final_Hs);
    final_vib.kernel();

    std::cout << "Initial-state zero-point harmonic vibrational energy = "
              << 0.5 * at::cat(init_vib.frequencies()).sum().item<double>() / 4.556335830019422e-6 << " cm^-1\n";
    for (size_t i = 0; i < init_vib.frequencies().size(); i++) {
        const at::Tensor & frequency = init_vib.frequencies()[i];
        std::ofstream ofs; ofs.open("initial-freq-" + std::to_string(i + 1) + ".txt");
        for (size_t j = 0; j < frequency.size(0); j++)
        ofs << std::scientific << std::setw(25) << std::setprecision(15) << frequency[j].item<double>() << '\n';
        ofs.close();
    }
    std::cout << "The initial-state normal modes can be visualized by init-*.log\n";
    for (size_t i = 0; i < init_vib.NIrreds(); i++) {
        auto freq = tchem::utility::tensor2vector(init_vib.frequencies()[i]);
        auto mode = tchem::utility::tensor2matrix(init_vib.cartmodes  ()[i]);
        CL::chem::xyz_vib<double> avogadro(init_geom.symbols(), init_geom.coords(), freq, mode, true);
        avogadro.print("init-" + std::to_string(i + 1) + ".log");
    }

    std::cout << "Final-state zero-point harmonic vibrational energy = "
              << 0.5 * at::cat(final_vib.frequencies()).sum().item<double>() / 4.556335830019422e-6 << " cm^-1\n";
    for (size_t i = 0; i < final_vib.frequencies().size(); i++) {
        const at::Tensor & frequency = final_vib.frequencies()[i];
        std::ofstream ofs; ofs.open("final-freq-" + std::to_string(i + 1) + ".txt");
        for (size_t j = 0; j < frequency.size(0); j++)
        ofs << std::scientific << std::setw(25) << std::setprecision(15) << frequency[j].item<double>() << '\n';
        ofs.close();
    }
    std::cout << "The final-state normal modes can be visualized by final-*.log\n";
    for (size_t i = 0; i < final_vib.NIrreds(); i++) {
        auto freq = tchem::utility::tensor2vector(final_vib.frequencies()[i]);
        auto mode = tchem::utility::tensor2matrix(final_vib.cartmodes  ()[i]);
        CL::chem::xyz_vib<double> avogadro(final_geom.symbols(), final_geom.coords(), freq, mode, true);
        avogadro.print("final-" + std::to_string(i + 1) + ".log");
    }
    std::cout << "The transformation matrix from internal coordiante to final-state normal coordinate can be found in Linv*.txt\n";
    for (size_t i = 0; i < final_vib.NIrreds(); i++) {
        const auto & Linv = final_vib.Linvs()[i];
        std::ofstream ofs;
        ofs.open("Linv-" + std::to_string(i + 1) + ".txt"); {
            for (size_t i = 0; i < Linv.size(0); i++) {
                for (size_t j = 0; j < Linv.size(1); j++)
                ofs << std::setw(25) << std::scientific << std::setprecision(15) << Linv[i][j].item<double>();
                ofs << '\n';
            }
        }
        ofs.close();
    }

    std::cout << '\n';
    final2init(init_qs, final_qs, init_vib, final_vib);

    std::cout << '\n';
    double contour = args.retrieve<double>("contour");
    suggest_phonons(contour, init_qs, final_qs, init_vib, final_vib);

    std::cout << '\n';
    int2normal(HdKernel, final_vib, final_qs[0]);

    std::cout << '\n';
    CL::utility::show_time(std::cout);
    std::cout << "Mission success\n";
}