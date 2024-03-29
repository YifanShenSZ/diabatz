# Find abinitio
# -------
#
# Finds abinitio
#
# This will define the following variables:
#
#   abinitio_FOUND        -- True if the system has abinitio
#   abinitio_INCLUDE_DIRS -- The include directories for abinitio
#   abinitio_LIBRARIES    -- Libraries to link against
#
# and the following imported targets:
#
#   abinitio

# Find abinitio root
# Assume we are in ${abinitioROOT}/share/cmake/abinitio/abinitioConfig.cmake
get_filename_component(CMAKE_CURRENT_LIST_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
get_filename_component(abinitioROOT "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

# include directory
set(abinitio_INCLUDE_DIRS ${abinitioROOT}/include)

# library
add_library(abinitio STATIC IMPORTED)
set(abinitio_LIBRARIES abinitio)

# dependency: Torch-Chemistry
if(NOT tchem_FOUND)
    find_package(tchem REQUIRED PATHS ~/Library/Torch-Chemistry)
    list(APPEND abinitio_INCLUDE_DIRS ${tchem_INCLUDE_DIRS})
    list(APPEND abinitio_LIBRARIES ${tchem_LIBRARIES})
    set(abinitio_CXX_FLAGS "${tchem_CXX_FLAGS}")
endif()

# import location
find_library(abinitio_LIBRARY abinitio PATHS "${abinitioROOT}/lib")
set_target_properties(abinitio PROPERTIES
    IMPORTED_LOCATION "${abinitio_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${abinitio_INCLUDE_DIRS}"
    CXX_STANDARD 14
)