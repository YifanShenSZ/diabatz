cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

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

include_directories(include)

add_library(Hderiva STATIC
    source/basic.cpp
    source/diabatic.cpp
    source/adiabatic.cpp
    source/composite.cpp
)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -qopenmp -parallel -ipo -static-intel")
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp")
endif ()

target_link_libraries(Hderiva tchem CL "${TORCH_LIBRARIES}")