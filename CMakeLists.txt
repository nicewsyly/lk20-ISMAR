cmake_minimum_required(VERSION 2.8)
add_definitions(-std=c++11)
set(CMAKE_MACOSX_RPATH 1)
project(lk20)
set(CMAKE_BUILD_TYPE DEBUG)
add_subdirectory(./toolslib libtools)
set(OpenCV_DIR /usr/local/lib/cmake/opencv4/)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} ./include ./toolslib/include)
file(GLOB SRC ./src/*.cpp)
add_executable(lk20 ${SRC})

target_link_libraries(lk20 libtools ${OpenCV_LIBS})

