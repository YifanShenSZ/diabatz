#include <CppLibrary/argparse.hpp>
#include <CppLibrary/chemistry.hpp>

#include <tchem/utility.hpp>
#include <tchem/intcoord.hpp>
#include <tchem/linalg.hpp>
#include <tchem/chemistry.hpp>

#include <Hd/kernel.hpp>

at::Tensor read_Columbus(const std::string & hessian_file, const tchem::IC::IntCoordSet & intcoordset);

void final2init(
const std::vector<size_t> & init_NModes, const std::vector<size_t> & final_NModes,
const at::Tensor & init_q, const at::Tensor & final_q,
const tchem::chem::SANormalMode & init_vib, const tchem::chem::SANormalMode & final_vib);

void suggest_phonons(const double & contour,
const std::vector<size_t> & init_NModes, const std::vector<size_t> & final_NModes,
const at::Tensor & init_q, const at::Tensor & final_q,
const tchem::chem::SANormalMode & init_vib, const tchem::chem::SANormalMode & final_vib);

void int2normal(const Hd::kernel & Hdkernel, const tchem::chem::SANormalMode & final_vib);

argparse::ArgumentParser parse_args(const size_t & argc, const char ** & argv) {
    CL::utility::echo_command(argc, argv, std::cout);
    std::cout << '\n';
    argparse::ArgumentParser parser("Generate *vibronics* input");

    // required arguments
    parser.add_argument("-d","--diabatz",  '+', false, "diabatz definition files");
    parser.add_argument("-f","--format",     1, false, "internal coordinate definition format (Columbus7, default)");
    parser.add_argument("-i","--IC",         1, false, "internal coordinate definition file");
    parser.add_argument("-m","--mass",       1, false, "the masses of atoms");
    parser.add_argument("--init_geom",       1, false, "intial-state equilibrium geometry");
    parser.add_argument("--final_geom",      1, false, " final-state equilibrium geometry");
    parser.add_argument("--init_NModes",   '+', false, "number of normal modes per intial-state irreducible");
    parser.add_argument("--final_NModes",  '+', false, "number of normal modes per  final-state irreducible");
    parser.add_argument("--init_hess",       1, false, "intial-state Columbus Hessian");
    parser.add_argument("-c","--contour",    1, false, "contour value");

    parser.parse_args(argc, argv);
    return parser;
}

at::Tensor compute_ddHd(const at::Tensor & r, const Hd::kernel & Hdkernel) {
    // Here ddHa is ▽[(▽H)a], computed by finite difference of (▽H)a
    const double dr = 1e-3;
    std::vector<at::Tensor> plus(r.size(0)), minus(r.size(0));
    #pragma omp parallel for
    for (size_t i = 0; i < r.size(0); i++) {
        at::Tensor Hd, dHd, energy, states;
        plus[i] = r.clone();
        plus[i][i] += dr;
        std::tie(Hd, plus[i]) = Hdkernel.compute_Hd_dHd(plus[i]);
        minus[i] = r.clone();
        minus[i][i] -= dr;
        std::tie(Hd, minus[i]) = Hdkernel.compute_Hd_dHd(minus[i]);
    }
    at::Tensor ddHd = r.new_empty({plus[0].size(0), plus[0].size(1), r.size(0), r.size(0)});
    for (size_t i = 0; i < r.size(0); i++) ddHd.select(2, i).copy_((plus[i] - minus[i]) / 2.0 / dr);
    return ddHd;
}

int main(size_t argc, const char ** argv) {
    std::cout << "Generate *vibronics* input\n"
              << "Yifan Shen 2021\n\n";
    argparse::ArgumentParser args = parse_args(argc, argv);
    CL::utility::show_time(std::cout);
    std::cout << '\n';

    std::vector<std::string> diabatz_inputs = args.retrieve<std::vector<std::string>>("diabatz");
    Hd::kernel Hdkernel(diabatz_inputs);

    std::string format = args.retrieve<std::string>("format");
    std::string IC     = args.retrieve<std::string>("IC");
    tchem::IC::IntCoordSet intcoordset(format, IC);

    std::string mass_file = args.retrieve<std::string>("mass"),
                final_geom_file = args.retrieve<std::string>("final_geom"),
                 init_geom_file = args.retrieve<std::string>( "init_geom");
    CL::chem::xyz_mass<double> final_geom(final_geom_file, mass_file, true),
                                init_geom( init_geom_file, mass_file, true);
    std::vector<double>  init_coords =  init_geom.coords(),
                        final_coords = final_geom.coords();
    at::Tensor  init_r = at::from_blob( init_coords.data(),  init_coords.size(), at::TensorOptions().dtype(torch::kFloat64)),
               final_r = at::from_blob(final_coords.data(), final_coords.size(), at::TensorOptions().dtype(torch::kFloat64));

    std::vector<size_t>  init_NModes = args.retrieve<std::vector<size_t>>( "init_NModes"),
                        final_NModes = args.retrieve<std::vector<size_t>>("final_NModes");
    size_t  init_NIrreds =  init_NModes.size(),
           final_NIrreds = final_NModes.size();
    if (std::accumulate(init_NModes.begin(), init_NModes.end(), 0) != intcoordset.size()) throw std::invalid_argument(
    "number of normal modes per intial-state irreducible must sum up to internal coordinate dimension");
    if (std::accumulate(final_NModes.begin(), final_NModes.end(), 0) != intcoordset.size()) throw std::invalid_argument(
    "number of normal modes per final-state irreducible must sum up to internal coordinate dimension");

    std::string init_hess_file = args.retrieve<std::string>("init_hess");
    at::Tensor init_H = read_Columbus(init_hess_file, intcoordset);
    at::Tensor init_q, init_J;
    std::tie(init_q, init_J) = intcoordset.compute_IC_J(init_r);
    std::vector<at::Tensor> init_Js(init_NIrreds), init_Hs(init_NIrreds);
    size_t start = 0;
    for (size_t i = 0; i < init_NIrreds; i++) {
        size_t stop = start + init_NModes[i];
        init_Js[i] = init_J.slice(0, start, stop);
        init_Hs[i] = init_H.slice(0, start, stop).slice(1, start, stop);
        start = stop;
    }
    tchem::chem::SANormalMode init_vib(init_geom.masses(), init_Js, init_Hs);
    init_vib.kernel();

    at::Tensor Hd, dHd;
    std::tie(Hd, dHd) = Hdkernel.compute_Hd_dHd(final_r);
    at::Tensor ddHd = compute_ddHd(final_r, Hdkernel);
    at::Tensor cartgrad = ( dHd[0][0] +  dHd[1][1]) / 2.0,
               carthess = (ddHd[0][0] + ddHd[1][1]) / 2.0;
    at::Tensor final_H = intcoordset.Hessian_cart2int(final_r, cartgrad, carthess);
    at::Tensor final_q, final_J;
    std::tie(final_q, final_J) = intcoordset.compute_IC_J(final_r);
    std::vector<at::Tensor> final_Js(final_NIrreds), final_Hs(final_NIrreds);
    start = 0;
    for (size_t i = 0; i < final_NIrreds; i++) {
        size_t stop = start + final_NModes[i];
        final_Js[i] = final_J.slice(0, start, stop);
        final_Hs[i] = final_H.slice(0, start, stop).slice(1, start, stop);
        start = stop;
    }
    tchem::chem::SANormalMode final_vib(final_geom.masses(), final_Js, final_Hs);
    final_vib.kernel();

    final2init(init_NModes, final_NModes, init_q, final_q, init_vib, final_vib);

    double contour = args.retrieve<double>("contour");
    suggest_phonons(contour, init_NModes, final_NModes, init_q, final_q, init_vib, final_vib);

    int2normal(Hdkernel, final_vib);

    std::cout << '\n';
    CL::utility::show_time(std::cout);
    std::cout << "Mission success\n";
}