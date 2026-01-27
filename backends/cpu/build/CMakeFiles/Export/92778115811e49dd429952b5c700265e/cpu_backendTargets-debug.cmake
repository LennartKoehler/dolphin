#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "cpu_backend::cpu_backend" for configuration "Debug"
set_property(TARGET cpu_backend::cpu_backend APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(cpu_backend::cpu_backend PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libcpu_backend.a"
  )

list(APPEND _cmake_import_check_targets cpu_backend::cpu_backend )
list(APPEND _cmake_import_check_files_for_cpu_backend::cpu_backend "${_IMPORT_PREFIX}/lib/libcpu_backend.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
