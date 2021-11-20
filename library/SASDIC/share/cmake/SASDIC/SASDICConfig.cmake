# Find SASDIC
# -------
#
# Finds SASDIC
#
# This will define the following variables:
#
#   SASDIC_FOUND        -- True if the system has SASDIC
#   SASDIC_INCLUDE_DIRS -- The include directories for SASDIC
#   SASDIC_LIBRARIES    -- Libraries to link against
#
# and the following imported targets:
#
#   SASDIC

# Find SASDIC root
# Assume we are in ${SASDICROOT}/share/cmake/SASDIC/SASDICConfig.cmake
get_filename_component(CMAKE_CURRENT_LIST_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
get_filename_component(SASDICROOT "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

# include directory
set(SASDIC_INCLUDE_DIRS ${SASDICROOT}/include)

# library
add_library(SASDIC STATIC IMPORTED)
set(SASDIC_LIBRARIES SASDIC)

# dependency 3: Torch-Chemistry
if(NOT tchem_FOUND)
    find_package(tchem REQUIRED PATHS ~/Library/Torch-Chemistry)
    list(APPEND SASDIC_INCLUDE_DIRS ${tchem_INCLUDE_DIRS})
    list(APPEND SASDIC_LIBRARIES ${tchem_LIBRARIES})
    set(SASDIC_CXX_FLAGS "${tchem_CXX_FLAGS}")
endif()

# dependency 2: Cpp-Library
if(NOT CL_FOUND)
    find_package(CL REQUIRED PATHS ~/Library/Cpp-Library)
    list(APPEND SASDIC_INCLUDE_DIRS ${CL_INCLUDE_DIRS})
    list(APPEND SASDIC_LIBRARIES ${CL_LIBRARIES})
endif()

# dependency 1: libtorch
if(NOT TORCH_FOUND)
    find_package(Torch REQUIRED PATHS ~/Software/Programming/libtorch) 
    list(APPEND SASDIC_INCLUDE_DIRS ${TORCH_INCLUDE_DIRS})
    list(APPEND SASDIC_LIBRARIES ${TORCH_LIBRARIES})
    set(SASDIC_CXX_FLAGS "${TORCH_CXX_FLAGS}")
endif()

# import location
find_library(SASDIC_LIBRARY SASDIC PATHS "${SASDICROOT}/lib")
set_target_properties(SASDIC PROPERTIES
    IMPORTED_LOCATION "${SASDIC_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${SASDIC_INCLUDE_DIRS}"
    CXX_STANDARD 14
)