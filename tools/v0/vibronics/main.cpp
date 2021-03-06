#include <GeometryTransformation.hpp>
#include <Chemistry.hpp>

#include <CppLibrary/argparse.hpp>
#include <CppLibrary/chemistry.hpp>

#include <tchem/intcoord.hpp>
#include <tchem/linalg.hpp>

#include <Hd/kernel.hpp>

argparse::ArgumentParser parse_args(const size_t & argc, const char ** & argv) {
    CL::utility::echo_command(argc, argv, std::cout);
    std::cout << '\n';
    argparse::ArgumentParser parser("Evaluation for diabatz version 0");

    // required arguments
    parser.add_argument("-f","--format",    1, false, "internal coordinate definition format (Columbus7, default)");
    parser.add_argument("-i","--IC",        1, false, "internal coordinate definition file");
    parser.add_argument("-m","--mass",      1, false, "the masses of atoms");
    parser.add_argument("-d","--diabatz", '+', false, "diabatz definition files");

    parser.parse_args(argc, argv);
    return parser;
}

at::Tensor compute_ddHd(const at::Tensor & r, const Hd::kernel & Hdkernel) {
    const double dr = 1e-3;
    std::vector<at::Tensor> plus(r.size(0)), minus(r.size(0));
    #pragma omp parallel for
    for (size_t i = 0; i < r.size(0); i++) {
        at::Tensor Hd;
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

    std::string format = args.retrieve<std::string>("format");
    std::string IC     = args.retrieve<std::string>("IC");
    tchem::IC::IntCoordSet intcoordset(format, IC);

    std::string mass_file = args.retrieve<std::string>("mass");
    CL::chem::xyz_mass<double> geom("mex.xyz", mass_file, true);

    std::vector<std::string> diabatz_inputs = args.retrieve<std::vector<std::string>>("diabatz");
    Hd::kernel Hdkernel(diabatz_inputs);

    std::vector<double> masses = geom.masses();
    std::vector<double> coords = geom.coords();
    at::Tensor r = at::from_blob(coords.data(), coords.size(), at::TensorOptions().dtype(torch::kFloat64));
    at::Tensor Hd, dHd;
    std::tie(Hd, dHd) = Hdkernel.compute_Hd_dHd(r);
    at::Tensor ddHd = compute_ddHd(r, Hdkernel);
    at::Tensor cartgrad = ( dHd[0][0] +  dHd[1][1]) / 2.0,
               carthess = (ddHd[0][0] + ddHd[1][1]) / 2.0;
    at::Tensor inthess = intcoordset.Hessian_cart2int(r, cartgrad, carthess);

    at::Tensor q, J;
    std::tie(q, J) = intcoordset.compute_IC_J(r);
    int32_t cartdim = r.size(0),
             intdim = q.size(0);
    int32_t NAtoms = cartdim / 3;
    at::Tensor JT = J.new_empty({cartdim, intdim});
    JT.copy_(J.transpose(0, 1));
    at::Tensor freq = q.new_empty(intdim),
               intmodeT  = J.new_empty({intdim,  intdim}),
               Linv      = J.new_empty({intdim,  intdim}),
               cartmodeT = J.new_empty({intdim, cartdim});
    FL::GT::WilsonGFMethod(inthess.data_ptr<double>(), JT.data_ptr<double>(), masses.data(),
                           freq.data_ptr<double>(), intmodeT.data_ptr<double>(), Linv.data_ptr<double>(),
                           cartmodeT.data_ptr<double>(), intdim, NAtoms);

    std::ofstream ofs; ofs.open("frequency.txt");
    for (int32_t i = 0; i < intdim; i++) ofs << freq[i].item<double>() << '\n';
    ofs.close();

    r /= 1.8897261339212517;
    freq /= 4.556335830019422e-6;
    // Wilson GF method normalizes Cartesian coordinate normal mode by Hessian metric
    // However, this may not be an appropriate magnitude to visualize
    // Here we use infinity-norm to normalize Cartesian coordinate normal mode
    // Actually, normalize to 9.99 since the visualization file format is %5.2f
    for (int32_t i = 0; i < intdim; i++)
    cartmodeT[i] *= 9.99 / at::amax(at::abs(cartmodeT[i]));
    FL::chem::Avogadro_Vibration(NAtoms, geom.symbols(), r.data_ptr<double>(), intdim,
                                 freq.data_ptr<double>(), cartmodeT.data_ptr<double>(),
                                 "mean-field.log");

    std::cout << '\n';
    CL::utility::show_time(std::cout);
    std::cout << "Mission success\n";
}