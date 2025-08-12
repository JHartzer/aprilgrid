set(DOXYGEN_EXTRACT_ALL            YES)
set(DOXYGEN_BUILTIN_STL_SUPPORT    YES)
set(DOXYGEN_JAVADOC_AUTOBRIEF      YES)
set(DOXYGEN_GENERATE_HTML          YES)
set(DOXYGEN_HAVE_DOT               YES)
set(DOXYGEN_USE_MDFILE_AS_MAINPAGE "${CMAKE_CURRENT_SOURCE_DIR}/../README.md")

doxygen_add_docs(documentation
    aprilgrid/aprilgrid.hpp
    ../README.md
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/include"
)