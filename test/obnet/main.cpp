#include <CppLibrary/chemistry.hpp>

#include <tchem/SASintcoord.hpp>
#include <tchem/SApolynomial.hpp>

#include <obnet/symat.hpp>

int main() {
    c10::TensorOptions top = at::TensorOptions().dtype(torch::kFloat64);
    size_t NStates = 2;

    tchem::IC::SASICSet sasicset("default", "IntCoordDef", "SAS.in");
    //std::cerr << sasicset.origin() << '\n';

    CL::utility::matrix<tchem::polynomial::SAPSet *> input_calculators(NStates);
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = i; j < NStates; j++)
    input_calculators[i][j] = new tchem::polynomial::SAPSet(std::to_string(i + 1) + std::to_string(j + 1) + ".in");

    obnet::symat Hd_net("net.in");
    Hd_net.to(torch::kFloat64);
    CL::utility::matrix<size_t> symmetry = Hd_net.symmetry();

    CL::chem::xyz<double> geom("min-C1.xyz", true);
    std::vector<double> coords = geom.coords();
    at::Tensor r = at::from_blob(coords.data(), coords.size(), top);
    at::Tensor q = sasicset.IntCoordSet::operator()(r);
    //std::cerr << q << '\n';

    std::vector<at::Tensor> xs = sasicset(q);
    CL::utility::matrix<at::Tensor> inputs(NStates);
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = i; j < NStates; j++)
    inputs[i][j] = (*input_calculators[i][j])(xs)[symmetry[i][j]];

    at::Tensor Hd = Hd_net(inputs);
    std::cout << Hd << '\n';
}