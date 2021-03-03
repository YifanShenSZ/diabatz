cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_BUILD_TYPE Release)

# Torch-Chemistry
set(CMAKE_PREFIX_PATH ~/Library/Torch-Chemistry)
find_package(tchem REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TCHEM_CXX_FLAGS}")

include_directories(include)

add_library(Hderiva STATIC
    source/basic.cpp
    source/diabatic.cpp
    source/adiabatic.cpp
    source/composite.cpp
)

target_link_libraries(Hderiva "${tchem_LIBRARIES}")