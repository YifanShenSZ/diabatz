cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(test)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_BUILD_TYPE Release)

# abinitio
set(CMAKE_PREFIX_PATH ~/Software/Mine/diabatz/library/abinitio)
find_package(abinitio REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${abinitio_CXX_FLAGS}")

# SASIC
set(CMAKE_PREFIX_PATH ~/Software/Mine/diabatz/library/SASIC)
find_package(SASIC REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SASIC_CXX_FLAGS}")

add_executable(test.exe global.cpp main.cpp)

target_link_libraries(test.exe ${SASIC_LIBRARIES} ${abinitio_LIBRARIES})