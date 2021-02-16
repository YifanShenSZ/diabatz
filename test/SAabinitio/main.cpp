#include <abinitio/SAreader.hpp>

#include "global.hpp"

int main() {
    sasicset = std::make_shared<tchem::IC::SASICSet>("default", "IntCoordDef", "SAS.in");

    std::vector<std::string> user_list = {"mex/", "min-C1/"};
    abinitio::SAReader reader(user_list, cart2int);
    reader.pretty_print(std::cout);

    std::shared_ptr<abinitio::DataSet<abinitio::SAGeometry>> GeomSet = reader.read_SAGeomSet();
    auto geom_loader = torch::data::make_data_loader(* GeomSet);
    size_t count = 0;
    for (auto & batch : * geom_loader)
    for (auto & data : batch) {
        auto qs = data->qs();
        for (auto & q : qs) std::cout << q.norm().item<double>() << ' ';
        std::cout << '\n';
        count++;
    }
    std::cout << "Number of geometries = " << count << ' '
              << GeomSet->size_int() << '\n';

    std::shared_ptr<abinitio::DataSet<abinitio::RegSAHam>> RegSet;
    std::shared_ptr<abinitio::DataSet<abinitio::DegSAHam>> DegSet;
    std::tie(RegSet, DegSet) = reader.read_SAHamSet();
    auto reg_loader = torch::data::make_data_loader(* RegSet);
    auto deg_loader = torch::data::make_data_loader(* DegSet);
    count = 0;
    for (auto & batch : * reg_loader)
    for (auto & data : batch) {
        for (size_t i = 0; i < data->energy().size(0); i++) {
            std::cout << data->energy()[i].item<double>() << "    ";
        }
        std::cout << '\n';
        count++;
    }
    std::cout << "Number of regular Hamiltonians = " << count << ' '
              << RegSet->size_int() << '\n';
    count = 0;
    for (auto & batch : * deg_loader)
    for (auto & data : batch) {
        std::cout << data->qs()[1].norm().item<double>() << '\n';
        std::cout << data->irreds()[0][1] << '\n';
        std::cout << data->dH()[0][1].size(0) << '\n';
        std::cout << data->H() << '\n';
        count++;
    }
    std::cout << "Number of degenerate Hamiltonians = " << count << ' '
              << DegSet->size_int() << '\n';
}