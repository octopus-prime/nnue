cmake_minimum_required(VERSION 3.28)

project(nnue CXX)

# set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -O3 -Wall")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(src)

# add_link_options(-flto)

file (GLOB_RECURSE DEMO CONFIGURE_DEPENDS ./demo/*.cpp ./demo/*.hpp)
file (GLOB_RECURSE TEST CONFIGURE_DEPENDS ./test/*.cpp ./test/*.hpp)

add_executable(nnue_demo ${DEMO})
add_executable(nnue_test ${TEST})
