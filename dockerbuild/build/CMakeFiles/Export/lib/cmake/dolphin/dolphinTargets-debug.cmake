#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "dolphin::dolphin" for configuration "Debug"
set_property(TARGET dolphin::dolphin APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(dolphin::dolphin PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libdolphin.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS dolphin::dolphin )
list(APPEND _IMPORT_CHECK_FILES_FOR_dolphin::dolphin "${_IMPORT_PREFIX}/lib/libdolphin.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
