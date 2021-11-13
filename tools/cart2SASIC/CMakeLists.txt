cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(cart2SASIC)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_BUILD_TYPE Release)

# SASIC
set(CMAKE_PREFIX_PATH ~/Software/Mine/diabatz/library/SASIC)
find_package(SASIC REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SASIC_CXX_FLAGS}")

add_executable(cart2SASIC.exe main.cpp)

target_link_libraries(cart2SASIC.exe ${SASIC_LIBRARIES})