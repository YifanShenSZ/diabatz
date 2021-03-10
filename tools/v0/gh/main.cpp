#include <CppLibrary/argparse.hpp>
#include <CppLibrary/chemistry.hpp>

#include <tchem/linalg.hpp>

#include <Hd/kernel.hpp>

argparse::ArgumentParser parse_args(const size_t & argc, const char ** & argv) {
    CL::utility::echo_command(argc, argv, std::cout);
    std::cout << '\n';
    argparse::ArgumentParser parser("Evaluation for diabatz version 0");

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

std::vector<at::Tensor> read_geoms(const std::string & file, const int64_t & NAtoms) {
    size_t NGeoms = CL::utility::NLines(file) / NAtoms;
    std::vector<at::Tensor> geoms(NGeoms);
    std::ifstream ifs;
    ifs.open(file);
    for (at::Tensor & geom : geoms) {
        geom = at::empty({NAtoms, 3}, c10::TensorOptions().dtype(torch::kFloat64));
        for (int64_t i = 0; i < NAtoms; i++) {
            std::string symbol;
            ifs >> symbol;
            double number;
            ifs >> number; geom[i][0] = number;
            ifs >> number; geom[i][1] = number;
            ifs >> number; geom[i][2] = number;
        }
        geom = geom.view(geom.numel());
    }
    ifs.close();
    return geoms;
}

void output_path(const std::string & prefix, const std::vector<at::Tensor> & energies, const std::vector<double> & nacs) {
    std::ofstream ofs;
    ofs.open(prefix + "-energy.txt");
    for (const at::Tensor & energy : energies) {
        for (size_t i = 0; i < energy.size(0); i++)
        ofs << energy[i].item<double>() << '\t';
        ofs << '\n';
    }
    ofs.close();
    ofs.open(prefix + "-nac.txt");
    for (const double & nac : nacs) ofs << nac << '\n';
    ofs.close();
}

int main(size_t argc, const char ** argv) {
    std::cout << "Evaluation for diabatz version 0\n"
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
    at::Tensor mex_geom = at::from_blob(mex_coords.data(), mex_coords.size(), at::TensorOptions().dtype(torch::kFloat64));
    at::Tensor mex_Hd, mex_dHd;
    std::tie(mex_Hd, mex_dHd) = Hdkernel.compute_Hd_dHd(mex_geom);
    at::Tensor mex_energy, mex_states;
    std::tie(mex_energy, mex_states) = mex_Hd.symeig(true);
    at::Tensor mex_dHa = tchem::linalg::UT_sy_U(mex_dHd, mex_states);
    double mex_nac = (mex_dHa[0][1] / (mex_energy[1] - mex_energy[0])).norm().item<double>();

    std::vector<at::Tensor> g_path_negative = read_geoms("g-path-negative.data", 18);
    std::vector<at::Tensor> g_path_positive = read_geoms("g-path-positive.data", 18);
    std::vector<at::Tensor> h_path_positive = read_geoms("h-path-positive.data", 18);

    std::vector<at::Tensor> energy_g_path(g_path_negative.size() + 1 + g_path_positive.size());
    std::vector<double>        nac_g_path(energy_g_path.size());
    for (size_t i = 0; i < g_path_negative.size(); i++) {
        at::Tensor Hd, dHd;
        std::tie(Hd, dHd) = Hdkernel.compute_Hd_dHd(g_path_negative[i]);
        at::Tensor energy, states;
        std::tie(energy, states) = Hd.symeig(true);
        at::Tensor dHa = tchem::linalg::UT_sy_U(dHd, states);
        double nac = (dHa[0][1] / (energy[1] - energy[0])).norm().item<double>();
        size_t index = g_path_negative.size() - 1 - i;
        energy_g_path[index] = energy.clone();
           nac_g_path[index] = nac;
    }
    energy_g_path[g_path_negative.size()] = mex_energy;
       nac_g_path[g_path_negative.size()] = mex_nac;
    for (size_t i = 0; i < g_path_positive.size(); i++) {
        at::Tensor Hd, dHd;
        std::tie(Hd, dHd) = Hdkernel.compute_Hd_dHd(g_path_positive[i]);
        at::Tensor energy, states;
        std::tie(energy, states) = Hd.symeig(true);
        at::Tensor dHa = tchem::linalg::UT_sy_U(dHd, states);
        double nac = (dHa[0][1] / (energy[1] - energy[0])).norm().item<double>();
        size_t index = g_path_negative.size() + 1 + i;
        energy_g_path[index] = energy.clone();
           nac_g_path[index] = nac;
    }

    std::vector<at::Tensor> energy_h_path(1 + 2 * h_path_positive.size());
    std::vector<double>        nac_h_path(energy_h_path.size());
    for (size_t i = 0; i < h_path_positive.size(); i++) {
        at::Tensor Hd, dHd;
        std::tie(Hd, dHd) = Hdkernel.compute_Hd_dHd(h_path_positive[i]);
        at::Tensor energy, states;
        std::tie(energy, states) = Hd.symeig(true);
        at::Tensor dHa = tchem::linalg::UT_sy_U(dHd, states);
        double nac = (dHa[0][1] / (energy[1] - energy[0])).norm().item<double>();
        size_t index = h_path_positive.size() + 1 + i;
        energy_h_path[index] = energy.clone();
           nac_h_path[index] = nac;
        index = h_path_positive.size() - 1 - i;
        energy_h_path[index] = energy.clone();
           nac_h_path[index] = nac;
    }
    energy_h_path[h_path_positive.size()] = mex_energy;
       nac_h_path[h_path_positive.size()] = mex_nac;

    output_path("g", energy_g_path, nac_g_path);
    output_path("h", energy_h_path, nac_h_path);

    std::cout << '\n';
    CL::utility::show_time(std::cout);
    std::cout << "Mission success\n";
}