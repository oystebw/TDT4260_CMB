cmake_minimum_required(VERSION 3.17)
project(tdt4260_optimization)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_STANDARD 99)

find_package(OpenCL)
find_package(OpenMP)

add_executable(checker checker.c ppm.c)

add_executable(image_processing_reference image_processing_reference.c ppm.c)
target_include_directories (image_processing_reference PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(image_processing_reference m)

add_executable(image_processing_c image_processing_c.c ppm.c)
target_include_directories (image_processing_c PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(image_processing_c m)
if(OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

if(OpenCL_FOUND)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DON_VM")
    link_directories(${OpenCL_LIBRARY})
    add_executable(image_processing_opencl image_processing_opencl.cpp ppm.c)
    target_include_directories (image_processing_opencl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${OpenCL_INCLUDE_DIRS})
    target_link_libraries(image_processing_opencl m ${OpenCL_LIBRARY})
endif()
