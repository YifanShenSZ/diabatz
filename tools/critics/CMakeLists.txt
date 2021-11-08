cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(critics)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_BUILD_TYPE Release)

# OpenMP
find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# Cpp-Library
set(CMAKE_PREFIX_PATH ~/Library/Cpp-Library)
find_package(CL REQUIRED)

# Foptim
set(CMAKE_PREFIX_PATH ~/Library/Foptim)
find_package(Foptim REQUIRED)

# Torch-Chemistry
set(CMAKE_PREFIX_PATH ~/Library/Torch-Chemistry)
find_package(tchem REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${tchem_CXX_FLAGS}")

# libHd
find_package(Hd REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Hd_CXX_FLAGS}")

include_directories(include)

add_executable(critics.exe
    source/utilities/fixed_intcoord.cpp
    source/utilities/int2cart.cpp
    source/utilities/Hd_extension.cpp
    source/utilities/global.cpp
    source/searchers/minimum.cpp
    source/searchers/saddle.cpp
    source/searchers/mex.cpp
    source/main.cpp
)

target_link_libraries(critics.exe
    ${Hd_LIBRARIES}
    ${tchem_LIBRARIES} ${Foptim_LIBRARIES} ${CL_LIBRARIES}
)