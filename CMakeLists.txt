cmake_minimum_required(VERSION 3.8)
project(RWdemo)

set(CMAKE_CXX_STANDARD 11)

SET(CMAKE_BUILD_TYPE Debug)

find_package(OpenCV REQUIRED)
find_package(GDAL REQUIRED)

include_directories(${GDAL_INCLUDE_DIRS})

set(FILES main.cpp )
add_executable(RWdemo ${FILES})


target_link_libraries(RWdemo ${OpenCV_LIBS} ${GDAL_LIBRARY})

