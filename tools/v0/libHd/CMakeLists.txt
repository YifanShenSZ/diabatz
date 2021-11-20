cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_BUILD_TYPE Release)

# Cpp-Library
set(CMAKE_PREFIX_PATH ~/Library/Cpp-Library)
find_package(CL REQUIRED)

# Torch-Chemistry
set(CMAKE_PREFIX_PATH ~/Library/Torch-Chemistry)
find_package(tchem REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TCHEM_CXX_FLAGS}")

# SASDIC
set(CMAKE_PREFIX_PATH ~/Software/Mine/diabatz/library/SASDIC)
find_package(SASDIC REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SASDIC_CXX_FLAGS}")

# obnet
set(CMAKE_PREFIX_PATH ~/Software/Mine/diabatz/library/obnet)
find_package(obnet REQUIRED)

# Hderiva
set(CMAKE_PREFIX_PATH ~/Software/Mine/diabatz/library/Hderiva)
find_package(Hderiva REQUIRED)

include_directories(include)

add_library(Hd STATIC
    source/InputGenerator.cpp
    source/kernel.cpp
)

target_link_libraries(Hd
    ${Hderiva_LIBRARIES} ${obnet_LIBRARIES}
    ${SASDIC_LIBRARIES}
    ${tchem_LIBRARIES} ${CL_LIBRARIES}
)