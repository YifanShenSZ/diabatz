cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

ENABLE_LANGUAGE(Fortran)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_BUILD_TYPE Release)

include_directories(include)

add_library(optim STATIC
    source/optim.f90
)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fpp")
else ()
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -cpp")
endif ()

target_link_libraries(optim "${TORCH_LIBRARIES}")