#include <abinitio/SAreader.hpp>

#include "global.hpp"

int main() {
    sasicset = std::make_shared<tchem::IC::SASICSet>("default", "IntCoordDef", "SAS.in");

    std::vector<std::string> user_list = {"mex-A1-B1/"};
    abinitio::SAReader reader(user_list, cart2CNPI);
    reader.pretty_print(std::cout);

    std::shared_ptr<abinitio::DataSet<abinitio::RegSAHam>> RegSet;
    std::shared_ptr<abinitio::DataSet<abinitio::DegSAHam>> DegSet;
    std::tie(RegSet, DegSet) = reader.read_SAHamSet();
    auto reg_loader = torch::data::make_data_loader(* RegSet);
    auto deg_loader = torch::data::make_data_loader(* DegSet);
    size_t count = 0;
    for (auto & batch : * reg_loader)
    for (auto & data : batch) {
        for (size_t i = 0; i < data->energy().size(0); i++)
        for (size_t j = i + 1; j < data->energy().size(0); j++) {
            std::cout << data->irreds(i, j) + 1 << "    ";
        }
        std::cout << '\n';
        count++;
    }
    std::cout << "Number of regular Hamiltonians = " << count << ' '
              << RegSet->size_int() << '\n';
    count = 0;
    for (auto & batch : * deg_loader)
    for (auto & data : batch) {
        std::cout << data->H() << '\n';
        count++;
    }
    std::cout << "Number of degenerate Hamiltonians = " << count << ' '
              << DegSet->size_int() << '\n';
}