cmake_minimum_required(VERSION 3.19)
project(part_2_benchmark CXX ISPC)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 20)

if(CMAKE_SIZEOF_VOID_P EQUAL 4)
  set(CMAKE_ISPC_FLAGS "--arch=x86")
endif()

add_library(part_2 OBJECT part_2.ispc)
set_target_properties(part_2 PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(part_2_benchmark "part_2_benchmark.cpp" "vector3.cpp")
target_link_libraries(part_2_benchmark PRIVATE part_2 picobench::picobench)
#set_target_properties(part_2_benchmark PROPERTIES FOLDER part_2)

#if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
#    target_compile_options(part_2_benchmark PRIVATE -msse4.2)
#elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
#    target_compile_options(part_2_benchmark PRIVATE /arch:AVX)
#endif()


