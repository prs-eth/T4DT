cmake_minimum_required(VERSION 3.22)
project(py_mepp2)

set(BUILD_USE_OPENMESH FALSE)
set(BUILD_USE_GUI FALSE)
set(BUILD_EXAMPLES FALSE)
set(BUILD_TESTS FALSE)

add_definitions(-DCGAL_DISABLE_ROUNDING_MATH_CHECK)

find_package(Eigen3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIRS})

add_subdirectory(external/pybind11)
add_subdirectory(external/MEPP2)

include_directories(external/MEPP2)
include_directories(external/MEPP2/External/CImg/CImg-2.4.2)

pybind11_add_module(py_mepp2 metrics.cpp)

target_compile_definitions(py_mepp2 PUBLIC)
