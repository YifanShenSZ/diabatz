# Find adiabatz
# -------
#
# Finds adiabatz
#
# This will define the following variables:
#
#   adiabatz_FOUND        -- True if the system has adiabatz
#   adiabatz_INCLUDE_DIRS -- The include directories for adiabatz
#   adiabatz_LIBRARIES    -- Libraries to link against
#
# and the following imported targets:
#
#   adiabatz

# Find adiabatz root
# Assume we are in ${adiabatzROOT}/share/cmake/adiabatz/adiabatzConfig.cmake
get_filename_component(CMAKE_CURRENT_LIST_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
get_filename_component(adiabatzROOT "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

# library
add_library(adiabatz STATIC IMPORTED)
set(adiabatz_LIBRARIES adiabatz)

# dependency 1: Torch-Chemistry
if(NOT tchem_FOUND)
    find_package(tchem REQUIRED PATHS ~/Library/Torch-Chemistry)
    list(APPEND adiabatz_INCLUDE_DIRS ${tchem_INCLUDE_DIRS})
    list(APPEND adiabatz_LIBRARIES ${tchem_LIBRARIES})
endif()

# dependency 2: Hd
if(NOT Hd_FOUND)
    find_package(Hd REQUIRED PATHS ~/Software/Mine/diabatz/tools/v0/libHd)
    list(APPEND adiabatz_INCLUDE_DIRS ${Hd_INCLUDE_DIRS})
    list(APPEND adiabatz_LIBRARIES ${Hd_LIBRARIES})
    set(adiabatz_CXX_FLAGS "${Hd_CXX_FLAGS}")
endif()

# import location
find_library(adiabatz_LIBRARY adiabatz PATHS "${adiabatzROOT}/lib")
set_target_properties(adiabatz PROPERTIES
    IMPORTED_LOCATION "${adiabatz_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${adiabatz_INCLUDE_DIRS}"
    CXX_STANDARD 14
)