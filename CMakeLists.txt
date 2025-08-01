cmake_minimum_required(VERSION 3.9)
set (CMAKE_CXX_STANDARD 17)
project(aprilgrid-cpp)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)
find_package(GTest REQUIRED)

include_directories(PUBLIC
	${OpenCV_INCLUDE_DIRS}
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
	$<INSTALL_INTERFACE:include>
)

add_library(aprilgrid-cpp
	src/aprilgrid.cpp
)

target_link_libraries(aprilgrid-cpp ${OpenCV_LIBS})
