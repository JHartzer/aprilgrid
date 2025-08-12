add_executable(aprilgrid-test aprilgrid_test.cpp)
target_compile_features(aprilgrid-test PUBLIC cxx_std_17)
target_link_libraries(aprilgrid-test ${OpenCV_LIBS} aprilgrid)

gtest_add_tests(aprilgrid-test "" AUTO)

target_link_libraries(
    aprilgrid-test
    GTest::gtest_main
)

enable_testing()
include(GoogleTest)
gtest_discover_tests(aprilgrid-test)