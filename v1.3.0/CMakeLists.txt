cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(diabatz)
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

# Torch-Chemistry
set(CMAKE_PREFIX_PATH ~/Library/Torch-Chemistry)
find_package(tchem REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TCHEM_CXX_FLAGS}")

# abinitio
set(CMAKE_PREFIX_PATH ~/Software/Mine/diabatz/library/abinitio)
find_package(abinitio REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${abinitio_CXX_FLAGS}")

# SASDIC
set(CMAKE_PREFIX_PATH ~/Software/Mine/diabatz/library/SASDIC)
find_package(SASDIC REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SASDIC_CXX_FLAGS}")

# obnet
set(CMAKE_PREFIX_PATH ~/Software/Mine/diabatz/library/obnet)
find_package(obnet REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${obnet_CXX_FLAGS}")

# Hderiva
set(CMAKE_PREFIX_PATH ~/Software/Mine/diabatz/library/Hderiva)
find_package(Hderiva REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Hderiva_CXX_FLAGS}")

# libHd
set(CMAKE_PREFIX_PATH ~/Software/Mine/diabatz/tools/v0/libHd)
find_package(Hd REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Hd_CXX_FLAGS}")

add_executable(diabatz.exe
    source/InputGenerator.cpp
    source/data_classes.cpp
    source/global.cpp
    source/data.cpp

    source/train/common.cpp
    source/train/residue.cpp
    source/train/Jacobian.cpp
    source/train/driver.cpp

    source/utility.cpp
    source/main.cpp
)

target_link_libraries(diabatz.exe
    ${Hd_LIBRARIES}
    ${Hderiva_LIBRARIES} ${obnet_LIBRARIES}
    ${SASDIC_LIBRARIES} ${abinitio_LIBRARIES}
    ${tchem_LIBRARIES} ${CL_LIBRARIES} ${Foptim_LIBRARIES}
    stdc++fs
)