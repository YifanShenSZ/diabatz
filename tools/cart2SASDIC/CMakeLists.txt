cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(cart2SASDIC)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_BUILD_TYPE Release)

# SASDIC
set(CMAKE_PREFIX_PATH ~/Software/Mine/diabatz/library/SASDIC)
find_package(SASDIC REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SASDIC_CXX_FLAGS}")

add_executable(cart2SASDIC.exe main.cpp)

target_link_libraries(cart2SASDIC.exe ${SASDIC_LIBRARIES})