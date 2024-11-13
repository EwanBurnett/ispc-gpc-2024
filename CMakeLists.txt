cmake_minimum_required(VERSION 3.19)
project(ispc_gpc_2024 C CXX)

# from ispc\examples\cpu\CMakeLists.txt
if(CMAKE_BUILD_TYPE)
    # Validate build type
    set(CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo")

    string(FIND "${CONFIGURATION_TYPES}" "${CMAKE_BUILD_TYPE}" MATCHED_CONFIG)
    if (${MATCHED_CONFIG} EQUAL -1)
        message(FATAL_ERROR "CMAKE_BUILD_TYPE (${CMAKE_BUILD_TYPE}) allows only the following values: ${CONFIGURATION_TYPES}")
    endif()
else(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
    message(STATUS "Build type not specified: Use Release by default.")
endif(CMAKE_BUILD_TYPE)

if(WIN32)
    add_compile_definitions(WIN32)
    if(MSVC)
        add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/EHsc>)
    endif()
endif()

if (NOT DEFINED ISPC_EXECUTABLE)
    find_program (ISPC_EXECUTABLE ispc)
    if (NOT ISPC_EXECUTABLE)
        message(FATAL_ERROR "Failed to find ispc" )
    endif()
endif()

add_definitions(-DNOMINMAX)

if("${CMAKE_SYSTEM_NAME};${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "Darwin;arm64")
  set(CMAKE_ISPC_INSTRUCTION_SETS "neon-i32x4")
else()
  set(CMAKE_ISPC_INSTRUCTION_SETS "sse2-i32x4;sse4-i32x4;avx1-i32x8;avx2-i32x8;avx512spr-x16")
endif()

add_library(ispc_gpc_2024 INTERFACE)
add_library(picobench::picobench ALIAS ispc_gpc_2024)
target_include_directories(ispc_gpc_2024 INTERFACE "external")

add_subdirectory(part_1)
add_subdirectory(part_2)
#add_subdirectory(rt)