cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_BUILD_TYPE Release)

# Torch-Chemistry
set(CMAKE_PREFIX_PATH ~/Library/Torch-Chemistry)
find_package(tchem REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${tchem_CXX_FLAGS}")

include_directories(include)

add_library(abinitio STATIC
    source/loader.cpp
    source/geometry.cpp
    source/energy.cpp
    source/Hamiltonian.cpp
    source/reader.cpp
    source/SAloader.cpp
    source/SAgeometry.cpp
    source/SAenergy.cpp
    source/SAHamiltonian.cpp
    source/SAreader.cpp
)

target_link_libraries(abinitio ${tchem_LIBRARIES})