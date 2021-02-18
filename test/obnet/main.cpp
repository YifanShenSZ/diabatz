#include <CppLibrary/chemistry.hpp>

#include <tchem/utility.hpp>
#include <tchem/SASintcoord.hpp>
#include <tchem/SApolynomial.hpp>

#include <obnet/symat.hpp>

#include "InputGenerator.hpp"

int main() {
    c10::TensorOptions top = at::TensorOptions().dtype(torch::kFloat64);
    size_t NStates = 2;

    tchem::IC::SASICSet sasicset("default", "IntCoordDef", "SAS.in");

    std::vector<std::string> sapoly_files = {"11.in", "12.in", "22.in"};
    InputGenerator input_generator(NStates,  sapoly_files);

    obnet::symat Hd_net("Hd.in");
    Hd_net.to(torch::kFloat64);

    CL::chem::xyz<double> geom("min-C1.xyz", true);
    std::vector<double> coords = geom.coords();
    at::Tensor r = at::from_blob(coords.data(), coords.size(), top);
    at::Tensor q = sasicset.IntCoordSet::operator()(r);
    std::vector<at::Tensor> qs = sasicset(q);
    CL::utility::matrix<at::Tensor> xs = input_generator(qs);

    at::Tensor Hd = Hd_net(xs);
    std::cout << "Hd =\n" << Hd << '\n';

    std::cout << "Number of parameters =\n"
              << tchem::utility::NParameters(Hd_net.elements->parameters())
              << '\n';
}