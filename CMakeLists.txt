cmake_minimum_required(VERSION 3.1)

project(freak_demo)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(freak_demo freak_demo.cpp)

target_link_libraries(freak_demo ${OpenCV_LIBS})