#include <CppLibrary/chemistry.hpp>

#include <tchem/utility.hpp>
#include <tchem/SASintcoord.hpp>
#include <tchem/SApolynomial.hpp>

#include <obnet/symat.hpp>

#include "global.hpp"

int main() {
    c10::TensorOptions top = at::TensorOptions().dtype(torch::kFloat64);

    sasicset = std::make_shared<tchem::IC::SASICSet>("default", "IntCoordDef", "SAS.in");

    Hdnet = std::make_shared<obnet::symat>("Hd.in");
    Hdnet->to(torch::kFloat64);

    std::vector<std::string> sapoly_files = {"11.in", "12.in", "22.in"};
    input_generator = std::make_shared<InputGenerator>(Hdnet->NStates(), Hdnet->irreds(), sapoly_files, sasicset->NSASICs());

    CL::chem::xyz<double> geom("min-C1.xyz", true);
    std::vector<double> coords = geom.coords();
    at::Tensor r = at::from_blob(coords.data(), coords.size(), top);
    at::Tensor q = sasicset->IntCoordSet::operator()(r);
    std::vector<at::Tensor> qs = (*sasicset)(q);
    CL::utility::matrix<at::Tensor> xs = (*input_generator)(qs);

    at::Tensor Hd = (*Hdnet)(xs);
    std::cout << "Hd =\n" << Hd << '\n';

    std::cout << "Number of parameters = "
              << tchem::utility::NParameters(Hdnet->elements->parameters())
              << '\n';
}