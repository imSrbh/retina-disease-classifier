cmake_minimum_required(VERSION 3.3)
project(retinaDiseaseClasifier)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package( OpenCV REQUIRED core highgui imgproc opencv_imgcodecs ml)

set(SOURCE_FILES main.cpp helpers.cpp helpers.h)
add_executable(retinaDiseaseClasifier ${SOURCE_FILES})

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(retinaDiseaseClasifier ${OpenCV_LIBS} )