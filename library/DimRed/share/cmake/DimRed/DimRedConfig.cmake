# Find DimRed
# -------
#
# Finds DimRed
#
# This will define the following variables:
#
#   DimRed_FOUND        -- True if the system has DimRed
#   DimRed_INCLUDE_DIRS -- The include directories for DimRed
#   DimRed_LIBRARIES    -- Libraries to link against
#
# and the following imported targets:
#
#   DimRed

# Find DimRed root
# Assume we are in ${DimRedROOT}/share/cmake/DimRed/DimRedConfig.cmake
get_filename_component(CMAKE_CURRENT_LIST_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
get_filename_component(DimRedROOT "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

# include directory
set(DimRed_INCLUDE_DIRS ${DimRedROOT}/include)

# library
add_library(DimRed STATIC IMPORTED)
set(DimRed_LIBRARIES DimRed)

# dependency: libtorch
if(NOT TORCH_FOUND)
    find_package(Torch REQUIRED PATHS ~/Software/Programming/libtorch) 
    list(APPEND DimRed_INCLUDE_DIRS ${TORCH_INCLUDE_DIRS})
    list(APPEND DimRed_LIBRARIES ${TORCH_LIBRARIES})
    set(DimRed_CXX_FLAGS "${TORCH_CXX_FLAGS}")
endif()

# import location
find_library(DimRed_LIBRARY DimRed PATHS "${DimRedROOT}/lib")
set_target_properties(DimRed PROPERTIES
    IMPORTED_LOCATION "${DimRed_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${DimRed_INCLUDE_DIRS}"
    CXX_STANDARD 14
)