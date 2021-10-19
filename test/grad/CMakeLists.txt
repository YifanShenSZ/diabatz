cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(test)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_BUILD_TYPE Release)

# libtorch
set(CMAKE_PREFIX_PATH ~/Software/Programming/libtorch)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Cpp-Library
include_directories(~/Library/Cpp-Library/include)
link_directories(~/Library/Cpp-Library/lib)

# Torch-Chemistry
include_directories(~/Library/Torch-Chemistry/include)
link_directories(~/Library/Torch-Chemistry/lib)

add_executable(test.exe 
    source/Hd.cpp
    source/commutor.cpp
    source/diabatic/DqHd.cpp
    source/diabatic/DcHd.cpp
    source/diabatic/DcDqHd.cpp
    source/adiabatic/DqHa.cpp
    source/adiabatic/DcHa.cpp
    source/adiabatic/Dc_DqH_a.cpp
    source/composite/composite.cpp
    source/composite/DcHc.cpp
    source/composite/Dc_DqH_c.cpp
    source/main.cpp
)

target_link_libraries(test.exe tchem CL "${TORCH_LIBRARIES}")
