# Findobnet
# -------
#
# Finds obnet
#
# This will define the following variables:
#
#   obnet_FOUND        -- True if the system has obnet
#   obnet_INCLUDE_DIRS -- The include directories for obnet
#   obnet_LIBRARIES    -- Libraries to link against
#
# and the following imported targets:
#
#   obnet

# Find obnet root
# Assume we are in ${obnetROOT}/share/cmake/obnet/obnetConfig.cmake
get_filename_component(CMAKE_CURRENT_LIST_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
get_filename_component(obnetROOT "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

# include directory
set(obnet_INCLUDE_DIRS ${obnetROOT}/include)

# library
add_library(obnet STATIC IMPORTED)
set(obnet_LIBRARIES obnet)

# import location
find_library(obnet_LIBRARY obnet PATHS "${obnetROOT}/lib")
set_target_properties(obnet PROPERTIES
    IMPORTED_LOCATION "${obnet_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${obnet_INCLUDE_DIRS}"
    CXX_STANDARD 14
)