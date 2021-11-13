#include <CppLibrary/argparse.hpp>
#include <CppLibrary/utility.hpp>

#include "../include/global.hpp"

void diabatic_obnet();

void diabatic_DimRed_obnet();

int main(const size_t & argc, const char ** & argv) {
    sasicset = std::make_shared<SASIC::SASICSet>("default", "IntCoordDef", "SAS.in");

    diabatic_obnet();
    std::cout << '\n';

    diabatic_DimRed_obnet();
}