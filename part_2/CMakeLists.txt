cmake_minimum_required(VERSION 3.19)
project(part_2_benchmark CXX ISPC)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 20)

if(CMAKE_SIZEOF_VOID_P EQUAL 4)
  set(CMAKE_ISPC_FLAGS "--arch=x86")
endif()

if("${CMAKE_SYSTEM_NAME};${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "Darwin;arm64")
  set(CMAKE_ISPC_INSTRUCTION_SETS "neon-i32x4")
else()
  set(CMAKE_ISPC_INSTRUCTION_SETS "sse2-i32x4;sse4-i32x4;avx1-i32x8;avx2-i32x8;avx512spr-x16")
endif()

add_library(part_2 OBJECT part_2.ispc)
set_target_properties(part_2 PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(part_2_benchmark "part_2_benchmark.cpp")
target_link_libraries(part_2_benchmark PRIVATE part_2 picobench::picobench)
set_target_properties(part_2_benchmark PROPERTIES FOLDER part_2)


