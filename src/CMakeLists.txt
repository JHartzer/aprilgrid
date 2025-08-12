
# This depends on OpenCV
find_package(OpenCV REQUIRED)

set(SRC_LIST aprilgrid.cpp)

set(HEADER_LIST "${aprilgrid_SOURCE_DIR}/include/aprilgrid/aprilgrid.hpp")

add_library(aprilgrid ${SRC_LIST} ${HEADER_LIST})

target_include_directories(aprilgrid PUBLIC ../include)

target_link_libraries(aprilgrid ${OpenCV_LIBS})

target_compile_features(aprilgrid PUBLIC cxx_std_98)

# include_directories(PUBLIC
#     ${OpenCV_INCLUDE_DIRS}
#     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#     $<INSTALL_INTERFACE:include>
# )

# install(
#     TARGETS aprilgrid
#     EXPORT AprilgridTargets
#     RUNTIME DESTINATION bin
#     LIBRARY DESTINATION lib
#     ARCHIVE DESTINATION lib
# )

# set(APRILGRID_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}/lib/cmake/aprilgrid/")

# install(
#     EXPORT AprilgridTargets
#     FILE AprilgridTargets.cmake
#     DESTINATION "${APRILGRID_INSTALL_PREFIX}"
#     NAMESPACE aprilgrid::
# )

# include(CMakePackageConfigHelpers)
# configure_package_config_file(
#     cmake/aprilgrid-config.cmake.in
#     "${CMAKE_CURRENT_BINARY_DIR}/aprilgrid-config.cmake"
#     INSTALL_DESTINATION "${APRILGRID_INSTALL_PREFIX}"
# )

# install(
#     FILES "${CMAKE_CURRENT_BINARY_DIR}/aprilgrid-config.cmake"
#     DESTINATION "${APRILGRID_INSTALL_PREFIX}"
# )

# install(FILES src/aprilgrid.hpp DESTINATION include)
