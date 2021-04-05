cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(gh)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_BUILD_TYPE Release)

# libHd
set(CMAKE_PREFIX_PATH ~/Software/Mine/diabatz/tools/v0/libHd)
find_package(Hd REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Hd_CXX_FLAGS}")

add_executable(gh.exe main.cpp)

target_link_libraries(gh.exe "${Hd_LIBRARIES}")
