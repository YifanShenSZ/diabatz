#include <CppLibrary/argparse.hpp>
#include <CppLibrary/chemistry.hpp>

#include <tchem/utility.hpp>
#include <tchem/intcoord.hpp>
#include <tchem/linalg.hpp>
#include <tchem/chemistry.hpp>

#include <Hd/kernel.hpp>

argparse::ArgumentParser parse_args(const size_t & argc, const char ** & argv) {
    CL::utility::echo_command(argc, argv, std::cout);
    std::cout << '\n';
    argparse::ArgumentParser parser("Generate *vibronics* input");

    // required arguments
    parser.add_argument("-f","--format",     1, false, "internal coordinate definition format (Columbus7, default)");
    parser.add_argument("-i","--IC",         1, false, "internal coordinate definition file");
    parser.add_argument("-1","--init_geom",  1, false, "intial state equilibrium geometry");
    parser.add_argument("-2","--final_geom", 1, false, " final state equilibrium geometry");
    parser.add_argument("-m","--mass",       1, false, "the masses of atoms");
    parser.add_argument("-d","--diabatz",  '+', false, "diabatz definition files");

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

at::Tensor compute_ddHa(const at::Tensor & r, const Hd::kernel & Hdkernel) {
    // Here ddHa is ▽[(▽H)a], computed by finite difference of (▽H)a
    const double dr = 1e-3;
    std::vector<at::Tensor> plus(r.size(0)), minus(r.size(0));
    #pragma omp parallel for
    for (size_t i = 0; i < r.size(0); i++) {
        at::Tensor Hd, dHd, energy, states;
        plus[i] = r.clone();
        plus[i][i] += dr;
        std::tie(Hd, dHd) = Hdkernel.compute_Hd_dHd(plus[i]);
        std::tie(energy, states) = Hd.symeig(true);
        plus[i] = tchem::linalg::UT_sy_U(dHd, states);
        minus[i] = r.clone();
        minus[i][i] -= dr;
        std::tie(Hd, dHd) = Hdkernel.compute_Hd_dHd(minus[i]);
        std::tie(energy, states) = Hd.symeig(true);
        minus[i] = tchem::linalg::UT_sy_U(dHd, states);
    }
    at::Tensor ddHa = r.new_empty({plus[0].size(0), plus[0].size(1), r.size(0), r.size(0)});
    for (size_t i = 0; i < r.size(0); i++) ddHa.select(2, i).copy_((plus[i] - minus[i]) / 2.0 / dr);
    return ddHa;
}

int main(size_t argc, const char ** argv) {
    std::cout << "Generate *vibronics* input\n"
              << "Yifan Shen 2021\n\n";
    argparse::ArgumentParser args = parse_args(argc, argv);
    CL::utility::show_time(std::cout);
    std::cout << '\n';

    std::string format = args.retrieve<std::string>("format");
    std::string IC     = args.retrieve<std::string>("IC");
    tchem::IC::IntCoordSet intcoordset(format, IC);

    std::string  init_geom_file = args.retrieve<std::string>( "init_geom"),
                final_geom_file = args.retrieve<std::string>("final_geom"),
                      mass_file = args.retrieve<std::string>("mass");
    CL::chem::xyz_mass<double>  init_geom( init_geom_file, mass_file, true),
                               final_geom(final_geom_file, mass_file, true);

    std::vector<std::string> diabatz_inputs = args.retrieve<std::vector<std::string>>("diabatz");
    Hd::kernel Hdkernel(diabatz_inputs);

    std::vector<double>  init_coords =  init_geom.coords(),
                        final_coords = final_geom.coords();
    at::Tensor r_init  = at::from_blob( init_coords.data(),  init_coords.size(), at::TensorOptions().dtype(torch::kFloat64)),
               r_final = at::from_blob(final_coords.data(), final_coords.size(), at::TensorOptions().dtype(torch::kFloat64));

    at::Tensor Hd, dHd, cartgrad, carthess;
    // initial state Hessian
    std::tie(Hd, dHd) = Hdkernel.compute_Hd_dHd(r_init);
    at::Tensor energy, states;
    std::tie(energy, states) = Hd.symeig(true);
    at::Tensor  dHa = tchem::linalg::UT_sy_U(dHd, states),
               ddHa = compute_ddHa(r_init, Hdkernel);
    cartgrad =  dHa[0][0];
    carthess = ddHa[0][0];
    at::Tensor init_inthess = intcoordset.Hessian_cart2int(r_init, cartgrad, carthess);
    // final state Hessian
    std::tie(Hd, dHd) = Hdkernel.compute_Hd_dHd(r_final);
    at::Tensor ddHd = compute_ddHa(r_final, Hdkernel);
    cartgrad = ( dHd[0][0] +  dHd[1][1]) / 2.0;
    carthess = (ddHd[0][0] + ddHd[1][1]) / 2.0;
    at::Tensor final_inthess = intcoordset.Hessian_cart2int(r_final, cartgrad, carthess);

    at::Tensor q, J;
    std::tie(q, J) = intcoordset.compute_IC_J(r_init);
    tchem::chem::IntNormalMode init_vib(init_geom.masses(), J, init_inthess);
    init_vib.kernel();
    std::tie(q, J) = intcoordset.compute_IC_J(r_final);
    tchem::chem::IntNormalMode final_vib(final_geom.masses(), J, final_inthess);
    final_vib.kernel();

    std::cerr << init_vib.frequency() / 4.556335830019422e-6 << '\n'
              << final_vib.frequency() / 4.556335830019422e-6 << '\n';

    std::cout << '\n';
    CL::utility::show_time(std::cout);
    std::cout << "Mission success\n";
}