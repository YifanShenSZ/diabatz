cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(CNPI2point)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_BUILD_TYPE Release)

# abinitio
set(CMAKE_PREFIX_PATH ~/Software/Mine/diabatz/library/abinitio)
find_package(abinitio REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${abinitio_CXX_FLAGS}")

add_executable(CNPI2point.exe
    global.cpp
    map.cpp
    main.cpp
)

target_link_libraries(CNPI2point.exe "${abinitio_LIBRARIES}")
