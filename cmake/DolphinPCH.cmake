# Shared precompiled-header configuration for the heavy dolphin targets.
#
# Nearly every TU in this project re-parses the same heavy headers
# (nlohmann/json via Config.h, spdlog, and the ITK image front-end via
# Image3D.h). target_precompile_headers() amortizes that parsing cost.
#
# Usage:
#   dolphin_target_pch(target)                   # full set (see below)
#   dolphin_target_pch(target HEADERS <h> ...)   # custom set for targets
#                                               # without ITK/nlohmann includes
#
# ccache note: PCH checksums and ccache can interact badly (cache misses or
# stale hits). If you build with CMAKE_*_COMPILER_LAUNCHER=ccache, configure
# the launcher as "ccache;-o;sloppiness=pch_defines,time_macros" or disable
# this option.

option(DOLPHIN_ENABLE_PCH "Enable precompiled headers for heavy targets" ON)

function(dolphin_target_pch target)
    if(NOT DOLPHIN_ENABLE_PCH)
        return()
    endif()
    if(NOT COMMAND target_precompile_headers)
        return()
    endif()

    set(headers
        <nlohmann/json.hpp>
        <spdlog/spdlog.h>
        <itkImage.h>
        <itkImageRegionIterator.h>
    )

    cmake_parse_arguments(ARG "" "" "HEADERS" ${ARGN})
    if(ARG_HEADERS)
        set(headers ${ARG_HEADERS})
    endif()

    target_precompile_headers(${target} PRIVATE ${headers})
endfunction()
