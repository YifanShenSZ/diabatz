cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(CNPI2point)
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

# libHd
include_directories(~/Software/Mine/diabatz/tools/v0/libHd/include)
link_directories(~/Software/Mine/diabatz/tools/v0/libHd/lib)

add_executable(RMSD.exe
    global.cpp
    compare.cpp
    main.cpp
)

target_link_libraries(RMSD.exe Hd Hderiva obnet abinitio tchem CL "${TORCH_LIBRARIES}")
