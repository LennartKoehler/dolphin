#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "dolphinbackend::dolphinbackend" for configuration "Debug"
set_property(TARGET dolphinbackend::dolphinbackend APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(dolphinbackend::dolphinbackend PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libdolphinbackend.a"
  )

list(APPEND _cmake_import_check_targets dolphinbackend::dolphinbackend )
list(APPEND _cmake_import_check_files_for_dolphinbackend::dolphinbackend "${_IMPORT_PREFIX}/lib/libdolphinbackend.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
