# FindHderiva
# -------
#
# Finds Hderiva
#
# This will define the following variables:
#
#   Hderiva_FOUND        -- True if the system has Hderiva
#   Hderiva_INCLUDE_DIRS -- The include directories for Hderiva
#   Hderiva_LIBRARIES    -- Libraries to link against
#
# and the following imported targets:
#
#   Hderiva

# Find Hderiva root
# Assume we are in ${HderivaROOT}/share/cmake/Hderiva/HderivaConfig.cmake
get_filename_component(CMAKE_CURRENT_LIST_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
get_filename_component(HderivaROOT "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

# include directory
set(Hderiva_INCLUDE_DIRS ${HderivaROOT}/include)

# library
add_library(Hderiva STATIC IMPORTED)
set(Hderiva_LIBRARIES Hderiva)

# dependency: Torch-Chemistry
if(NOT tchem_FOUND)
    find_package(tchem REQUIRED PATHS ~/Library/Torch-Chemistry)
    list(APPEND Hderiva_INCLUDE_DIRS ${tchem_INCLUDE_DIRS})
    list(APPEND Hderiva_LIBRARIES ${tchem_LIBRARIES})
    set(Hderiva_CXX_FLAGS "${tchem_CXX_FLAGS}")
endif()

# import location
find_library(Hderiva_LIBRARY Hderiva PATHS "${HderivaROOT}/lib")
set_target_properties(Hderiva PROPERTIES
    IMPORTED_LOCATION "${Hderiva_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${Hderiva_INCLUDE_DIRS}"
    CXX_STANDARD 14
)