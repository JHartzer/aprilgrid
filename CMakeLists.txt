cmake_minimum_required(VERSION 3.9)
set (CMAKE_CXX_STANDARD 17)
project(AprilGrid)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)
find_package(GTest REQUIRED)

include_directories(PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)

add_library(aprilgrid
    src/aprilgrid.cpp
)

target_link_libraries(aprilgrid ${OpenCV_LIBS})

install(
    TARGETS aprilgrid
    EXPORT AprilgridTargets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

set(APRILGRID_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}/lib/cmake/aprilgrid/")

install(
    EXPORT AprilgridTargets
    FILE AprilgridTargets.cmake
    DESTINATION "${APRILGRID_INSTALL_PREFIX}"
    NAMESPACE aprilgrid::
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    cmake/aprilgrid-config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/aprilgrid-config.cmake"
    INSTALL_DESTINATION "${APRILGRID_INSTALL_PREFIX}"
)

install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/aprilgrid-config.cmake"
    DESTINATION "${APRILGRID_INSTALL_PREFIX}"
)

install(FILES src/aprilgrid.hpp DESTINATION include)

add_executable(aprilgrid-test src/test/aprilgrid_test.cpp)
target_link_libraries(aprilgrid-test aprilgrid ${OpenCV_LIBS})

gtest_add_tests(aprilgrid-test "" AUTO)

target_link_libraries(
    aprilgrid-test
    GTest::gtest_main
)

enable_testing()
include(GoogleTest)
gtest_discover_tests(aprilgrid-test)
