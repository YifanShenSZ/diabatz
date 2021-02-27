cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(txt2bin)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_BUILD_TYPE Release)

# libtorch
set(CMAKE_PREFIX_PATH ~/Software/Programming/libtorch-cuda10.1-1.7.1)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Cpp-Library
include_directories(~/Library/Cpp-Library/include)
link_directories(~/Library/Cpp-Library/lib)

# obnet
include_directories(~/Software/Mine/diabatz/library/obnet/include)
link_directories(~/Software/Mine/diabatz/library/obnet/lib)

add_executable(txt2bin.exe main.cpp)

target_link_libraries(txt2bin.exe obnet CL "${TORCH_LIBRARIES}")
