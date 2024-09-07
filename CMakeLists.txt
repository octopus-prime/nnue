cmake_minimum_required(VERSION 3.28)

project(nnue CXX)

set(CMAKE_CXX_COMPILER g++-14)
set(CMAKE_C_COMPILER gcc-14)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
set(CMAKE_CXX_STANDARD 26)
# set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -Wall -Wextra -O3 -ffast-math -funroll-loops")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -Wall -Wextra -g")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_link_options(-flto)

file (GLOB_RECURSE SOURCES CONFIGURE_DEPENDS ./src/*.cpp ./src/*.hpp)

add_executable(nnue ${SOURCES})
