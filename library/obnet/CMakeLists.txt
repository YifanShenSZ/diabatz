cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_BUILD_TYPE Release)

# libtorch
set(CMAKE_PREFIX_PATH ~/Software/Programming/libtorch-cuda10.1-1.7.1)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Cpp-Library
set(CMAKE_PREFIX_PATH ~/Library/Cpp-Library)
find_package(CL REQUIRED)

include_directories(include)

add_library(obnet STATIC
    source/scalar.cpp
    source/symat.cpp
)

target_link_libraries(obnet CL "${TORCH_LIBRARIES}")