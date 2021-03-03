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

# dependency 1: Torch-Chemistry
find_package(tchem REQUIRED PATHS ~/Library/Torch-Chemistry)
list(APPEND Hd_INCLUDE_DIRS ${tchem_INCLUDE_DIRS})
list(APPEND Hd_LIBRARIES ${tchem_LIBRARIES})
set(Hd_CXX_FLAGS "${tchem_CXX_FLAGS}")

# dependency 2: obnet
find_package(obnet REQUIRED PATHS ~/Software/Mine/diabatz/library/obnet)
list(APPEND Hd_INCLUDE_DIRS ${obnet_INCLUDE_DIRS})
list(APPEND Hd_LIBRARIES ${obnet_LIBRARIES})

# dependency 3: Hderiva
find_package(Hderiva REQUIRED PATHS ~/Software/Mine/diabatz/library/Hderiva)
list(APPEND Hd_INCLUDE_DIRS ${Hderiva_INCLUDE_DIRS})
list(APPEND Hd_LIBRARIES ${Hderiva_LIBRARIES})

# import location
find_library(Hd_LIBRARY Hd PATHS "${HdROOT}/lib")
set_target_properties(Hd PROPERTIES
    IMPORTED_LOCATION "${Hd_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${Hd_INCLUDE_DIRS}"
    CXX_STANDARD 14
)