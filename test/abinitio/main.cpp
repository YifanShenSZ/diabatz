#include <abinitio/reader.hpp>

int main() {
    std::vector<std::string> user_list = {"mex-A1-B1/", "list"};
    abinitio::Reader reader(user_list);
    reader.pretty_print(std::cout);

    abinitio::DataSet<abinitio::Geometry> * GeomSet = reader.read_GeomSet();
    auto geom_loader = torch::data::make_data_loader(* GeomSet);
    size_t count = 0;
    for (auto & batch : * geom_loader)
    for (auto & data : batch) {
        std::cout << data->geom() << '\n';
        count++;
    }
    std::cout << "Number of geometries = " << count << ' '
              << GeomSet->size_int() << '\n';

    abinitio::DataSet<abinitio::RegHam> * RegSet;
    abinitio::DataSet<abinitio::DegHam> * DegSet;
    std::tie(RegSet, DegSet) = reader.read_HamSet();
    auto reg_loader = torch::data::make_data_loader(* RegSet);
    auto deg_loader = torch::data::make_data_loader(* DegSet);
    count = 0;
    for (auto & batch : * reg_loader)
    for (auto & data : batch) {
        std::cout << data->energy() << '\n';
        count++;
    }
    std::cout << "Number of regular Hamiltonians = " << count << ' '
              << RegSet->size_int() << '\n';
    count = 0;
    for (auto & batch : * deg_loader)
    for (auto & data : batch) {
        std::cout << data->energy() << '\n';
        count++;
    }
    std::cout << "Number of degenerate Hamiltonians = " << count << ' '
              << DegSet->size_int() << '\n';
}