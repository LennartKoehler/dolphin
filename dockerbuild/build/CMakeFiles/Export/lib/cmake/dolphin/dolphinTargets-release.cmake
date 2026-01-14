#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "dolphin::dolphin" for configuration "Release"
set_property(TARGET dolphin::dolphin APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(dolphin::dolphin PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libdolphin.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS dolphin::dolphin )
list(APPEND _IMPORT_CHECK_FILES_FOR_dolphin::dolphin "${_IMPORT_PREFIX}/lib/libdolphin.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
