#include <abinitio/SAreader.hpp>

#include "global.hpp"

int main() {
    sasicset = std::make_shared<tchem::IC::SASICSet>("default", "IntCoordDef", "SAS.in");

    abinitio::SAReader reader({"min-B1/"}, cart2CNPI);
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
    count = 0;
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
}