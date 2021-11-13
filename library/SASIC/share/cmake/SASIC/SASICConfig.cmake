# Find SASIC
# -------
#
# Finds SASIC
#
# This will define the following variables:
#
#   SASIC_FOUND        -- True if the system has SASIC
#   SASIC_INCLUDE_DIRS -- The include directories for SASIC
#   SASIC_LIBRARIES    -- Libraries to link against
#
# and the following imported targets:
#
#   SASIC

# Find SASIC root
# Assume we are in ${SASICROOT}/share/cmake/SASIC/SASICConfig.cmake
get_filename_component(CMAKE_CURRENT_LIST_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
get_filename_component(SASICROOT "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

# include directory
set(SASIC_INCLUDE_DIRS ${SASICROOT}/include)

# library
add_library(SASIC STATIC IMPORTED)
set(SASIC_LIBRARIES SASIC)

# dependency 3: Torch-Chemistry
if(NOT tchem_FOUND)
    find_package(tchem REQUIRED PATHS ~/Library/Torch-Chemistry)
    list(APPEND SASIC_INCLUDE_DIRS ${tchem_INCLUDE_DIRS})
    list(APPEND SASIC_LIBRARIES ${tchem_LIBRARIES})
    set(SASIC_CXX_FLAGS "${tchem_CXX_FLAGS}")
endif()

# dependency 2: Cpp-Library
if(NOT CL_FOUND)
    find_package(CL REQUIRED PATHS ~/Library/Cpp-Library)
    list(APPEND SASIC_INCLUDE_DIRS ${CL_INCLUDE_DIRS})
    list(APPEND SASIC_LIBRARIES ${CL_LIBRARIES})
endif()

# dependency 1: libtorch
if(NOT TORCH_FOUND)
    find_package(Torch REQUIRED PATHS ~/Software/Programming/libtorch) 
    list(APPEND SASIC_INCLUDE_DIRS ${TORCH_INCLUDE_DIRS})
    list(APPEND SASIC_LIBRARIES ${TORCH_LIBRARIES})
    set(SASIC_CXX_FLAGS "${TORCH_CXX_FLAGS}")
endif()

# import location
find_library(SASIC_LIBRARY SASIC PATHS "${SASICROOT}/lib")
set_target_properties(SASIC PROPERTIES
    IMPORTED_LOCATION "${SASIC_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${SASIC_INCLUDE_DIRS}"
    CXX_STANDARD 14
)