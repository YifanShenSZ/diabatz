cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_BUILD_TYPE Release)

# libtorch
set(CMAKE_PREFIX_PATH ~/Software/Programming/libtorch)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(include)

add_library(DimRed STATIC
    source/encoder.cpp
    source/decoder.cpp
)

target_link_libraries(DimRed "${TORCH_LIBRARIES}")