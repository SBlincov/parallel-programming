cmake_minimum_required(VERSION 3.10)
project(blintsov_sd)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_COMPILER "/Users/sblincov/opt/usr/local/bin/mpicxx")
find_package ( MPI )
if ( MPI_FOUND )
    include_directories( ${MPI_INCLUDE_PATH} )
endif( MPI_FOUND )
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(blintsov_sd lab3/main.cpp)

target_link_libraries(blintsov_sd ${OpenCV_LIBS})