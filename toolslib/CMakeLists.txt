cmake_minimum_required(VERSION 2.8)
set(CMAKE_BUILD_TYPE DEBUG)
project(libtools)
set(OpenCV_DIR /usr/local/lib/cmake/opencv4/)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(./include)

file(GLOB SRC "./src/*.cpp")
add_library(libtools SHARED ${SRC})
target_link_libraries(libtools ${OpenCV_LIBS})

