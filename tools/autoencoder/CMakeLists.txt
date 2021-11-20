cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(autoencoder)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_BUILD_TYPE Release)

# OpenMP
find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# Foptim
set(CMAKE_PREFIX_PATH ~/Library/Foptim)
find_package(Foptim REQUIRED)

# Cpp-Library
set(CMAKE_PREFIX_PATH ~/Library/Cpp-Library)
find_package(CL REQUIRED)

# abinitio
set(CMAKE_PREFIX_PATH ~/Software/Mine/diabatz/library/abinitio)
find_package(abinitio REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${abinitio_CXX_FLAGS}")

# SASDIC
set(CMAKE_PREFIX_PATH ~/Software/Mine/diabatz/library/SASDIC)
find_package(SASDIC REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SASDIC_CXX_FLAGS}")

# DimRed
set(CMAKE_PREFIX_PATH ~/Software/Mine/diabatz/library/DimRed)
find_package(DimRed REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${DimRed_CXX_FLAGS}")

add_executable(autoencoder.exe
    source/global.cpp
    source/Adam.cpp
    source/SGD.cpp
    source/trust_region.cpp
    source/main.cpp
)

target_link_libraries(autoencoder.exe
    ${DimRed_LIBRARIES} ${SASDIC_LIBRARIES} ${abinitio_LIBRARIES}
    ${CL_LIBRARIES} ${Foptim_LIBRARIES}
)