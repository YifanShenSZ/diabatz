cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(test)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_BUILD_TYPE Release)

# OpenMP
find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# abinitio
set(CMAKE_PREFIX_PATH ~/Software/Mine/diabatz/library/abinitio)
find_package(abinitio REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${abinitio_CXX_FLAGS}")

# SASIC
set(CMAKE_PREFIX_PATH ~/Software/Mine/diabatz/library/SASIC)
find_package(SASIC REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SASIC_CXX_FLAGS}")

# DimRed
set(CMAKE_PREFIX_PATH ~/Software/Mine/diabatz/library/DimRed)
find_package(DimRed REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${DimRed_CXX_FLAGS}")

# obnet
set(CMAKE_PREFIX_PATH ~/Software/Mine/diabatz/library/obnet)
find_package(obnet REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${obnet_CXX_FLAGS}")

# Hderiva
set(CMAKE_PREFIX_PATH ~/Software/Mine/diabatz/library/Hderiva)
find_package(Hderiva REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Hderiva_CXX_FLAGS}")

add_executable(test.exe
    source/InputGenerator.cpp
    source/global.cpp
    source/data.cpp
    source/diabatic_obnet.cpp
    source/diabatic_DimRed-obnet.cpp
    source/main.cpp
)

target_link_libraries(test.exe
    ${Hderiva_LIBRARIES}
    ${obnet_LIBRARIES}
    ${DimRed_LIBRARIES}
    ${SASIC_LIBRARIES}
    ${abinitio_LIBRARIES}
)