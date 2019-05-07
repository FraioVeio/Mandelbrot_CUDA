cmake_minimum_required(VERSION 2.8)

project(Mandelbrot)

SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CUDA_STANDARD 11)


find_package(CUDA REQUIRED)
find_package( OpenCV REQUIRED )

list( APPEND CUDA_NVCC_FLAGS "-std=c++11")

include_directories(${OpenCV_INCLUDE_DIRS})

cuda_add_executable(Mandelbrot mandelbrot.cu)
target_link_libraries (Mandelbrot ${OpenCV_LIBS} m)
