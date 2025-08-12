cmake_minimum_required(VERSION 3.9)

project(
    aprilgrid
    VERSION 0.1
    DESCRIPTION "A basic C++ library for performing april grid detections and pose estimations using OpenCV"
    LANGUAGES CXX
)

# Only do these if this is the main project, and not if it is included through add_subdirectory
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)

    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)

    # Build tests
    find_package(GTest REQUIRED)
    if(GTest_FOUND)
        add_subdirectory(test)
    else()
        message(STATUS "GTest not found, not building tests")
    endif()

    # Build docs
    find_package(Doxygen)
    if(Doxygen_FOUND)
        set(DOXYGEN_EXTRACT_ALL            YES)
        set(DOXYGEN_BUILTIN_STL_SUPPORT    YES)
        set(DOXYGEN_JAVADOC_AUTOBRIEF      YES)
        set(DOXYGEN_GENERATE_HTML          YES)
        set(DOXYGEN_HAVE_DOT               YES)
        set(DOXYGEN_IMAGE_PATH             "${PROJECT_SOURCE_DIR}/test/assets")
        set(DOXYGEN_USE_MDFILE_AS_MAINPAGE "${CMAKE_CURRENT_SOURCE_DIR}/README.md")

        doxygen_add_docs(documentation
            aprilgrid/aprilgrid.hpp
            "${CMAKE_CURRENT_SOURCE_DIR}/README.md"
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/include"
        )
    else()
        message(STATUS "Doxygen not found, not building docs")
    endif()
else()
    message(STATUS "aprilgrid not primary project")
endif()

add_subdirectory(src)
