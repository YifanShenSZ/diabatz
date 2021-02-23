cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(v0)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_BUILD_TYPE Release)

# OpenMP
find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# libtorch
set(CMAKE_PREFIX_PATH ~/Software/Programming/libtorch-cuda10.1-1.7.1)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Cpp-Library
include_directories(~/Library/Cpp-Library/include)
link_directories(~/Library/Cpp-Library/lib)

# Torch-Chemistry
include_directories(~/Library/Torch-Chemistry/include)
link_directories(~/Library/Torch-Chemistry/lib)

# abinitio
include_directories(~/Software/Mine/diabatz/library/abinitio/include)
link_directories(~/Software/Mine/diabatz/library/abinitio/lib)

# obnet
include_directories(~/Software/Mine/diabatz/library/obnet/include)
link_directories(~/Software/Mine/diabatz/library/obnet/lib)

# Hderiva
include_directories(~/Software/Mine/diabatz/library/Hderiva/include)
link_directories(~/Software/Mine/diabatz/library/Hderiva/lib)

# Fopt
include_directories(~/Software/Mine/diabatz/library/Fopt/include)
link_directories(~/Software/Mine/diabatz/library/Fopt/lib)

add_executable(v0.exe
    InputGenerator.cpp
    global.cpp
    data.cpp
    residue.cpp
    Jacobian.cpp
    train.cpp
    main.cpp
)

target_link_libraries(v0.exe
ifcore mkl_intel_lp64 mkl_intel_thread mkl_core
Fopt Hderiva obnet abinitio tchem CL "${TORCH_LIBRARIES}")