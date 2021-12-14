# FindHd
# -------
#
# Finds Hd
#
# This will define the following variables:
#
#   Hd_FOUND        -- True if the system has Hd
#   Hd_INCLUDE_DIRS -- The include directories for Hd
#   Hd_LIBRARIES    -- Libraries to link against
#   Hd_CXX_FLAGS    -- Additional (required) compiler flags
#
# and the following imported targets:
#
#   Hd

# Find Hd root
# Assume we are in ${HdROOT}/share/cmake/Hd/HdConfig.cmake
get_filename_component(CMAKE_CURRENT_LIST_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
get_filename_component(HdROOT "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

# include directory
set(Hd_INCLUDE_DIRS ${HdROOT}/include)

# library
add_library(Hd STATIC IMPORTED)
set(Hd_LIBRARIES Hd)

# dependency 5: Hderiva
if(NOT Hderiva_FOUND)
    find_package(Hderiva REQUIRED PATHS ~/Software/Mine/diabatz/library/Hderiva)
    list(APPEND Hd_INCLUDE_DIRS ${Hderiva_INCLUDE_DIRS})
    list(APPEND Hd_LIBRARIES ${Hderiva_LIBRARIES})
    set(Hd_CXX_FLAGS "${Hderiva_CXX_FLAGS}")
endif()

# dependency 4: obnet
if(NOT obnet_FOUND)
    find_package(obnet REQUIRED PATHS ~/Software/Mine/diabatz/library/obnet)
    list(APPEND Hd_INCLUDE_DIRS ${obnet_INCLUDE_DIRS})
    list(APPEND Hd_LIBRARIES ${obnet_LIBRARIES})
    set(Hd_CXX_FLAGS "${obnet_CXX_FLAGS}")
endif()

# dependency 3: SASDIC
if(NOT SASDIC_FOUND)
    find_package(SASDIC REQUIRED PATHS ~/Software/Mine/diabatz/library/SASDIC)
    list(APPEND Hd_INCLUDE_DIRS ${SASDIC_INCLUDE_DIRS})
    list(APPEND Hd_LIBRARIES ${SASDIC_LIBRARIES})
    set(Hd_CXX_FLAGS "${SASDIC_CXX_FLAGS}")
endif()

# dependency 2: Torch-Chemistry
if(NOT tchem_FOUND)
    find_package(tchem REQUIRED PATHS ~/Library/Torch-Chemistry)
    list(APPEND Hd_INCLUDE_DIRS ${tchem_INCLUDE_DIRS})
    list(APPEND Hd_LIBRARIES ${tchem_LIBRARIES})
    set(Hd_CXX_FLAGS "${tchem_CXX_FLAGS}")
endif()

# dependency 1: Cpp-Library
if(NOT CL_FOUND)
    find_package(CL REQUIRED PATHS ~/Library/Cpp-Library)
    list(APPEND Hd_INCLUDE_DIRS ${CL_INCLUDE_DIRS})
    list(APPEND Hd_LIBRARIES ${CL_LIBRARIES})
endif()

# import location
find_library(Hd_LIBRARY Hd PATHS "${HdROOT}/lib")
set_target_properties(Hd PROPERTIES
    IMPORTED_LOCATION "${Hd_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${Hd_INCLUDE_DIRS}"
    CXX_STANDARD 14
)