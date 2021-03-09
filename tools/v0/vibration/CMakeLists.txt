cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(vibration)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_BUILD_TYPE Release)

# OpenMP
find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# Fortran-Library
set(CMAKE_PREFIX_PATH ~/Library/Fortran-Library)
find_package(FL REQUIRED)

# libHd
set(CMAKE_PREFIX_PATH ~/Software/Mine/diabatz/tools/v0/libHd)
find_package(Hd REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Hd_CXX_FLAGS}")

add_executable(vibration.exe main.cpp)

target_link_libraries(vibration.exe "${Hd_LIBRARIES}" "${FL_LIBRARIES}")
