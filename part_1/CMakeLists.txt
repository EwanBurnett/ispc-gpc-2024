cmake_minimum_required(VERSION 3.19)
project(part_1_benchmark CXX ISPC)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 20)

if(CMAKE_SIZEOF_VOID_P EQUAL 4)
  set(CMAKE_ISPC_FLAGS "--arch=x86")
endif()

add_library(part_1 OBJECT part_1.ispc)
set_target_properties(part_1 PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(part_1_benchmark "part_1_benchmark.cpp")
target_link_libraries(part_1_benchmark PRIVATE part_1 picobench::picobench)
#set_target_properties(part_1_benchmark PROPERTIES FOLDER part_1)

