

#include "aprilgrid.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace fs = std::filesystem;

class AprilgridDetectorTest : public testing::Test {
 protected:
  cv::Mat camera_matrix_ = (cv::Mat_<float>(3, 3) << 500, 0, 512, 0, 500, 512, 0, 0, 0);
  cv::Mat dist_coeffs_{0, 0, 0, 0};
};

TEST_F(AprilgridDetectorTest, full) {
  auto detector = AprilGrid(cv::aruco::DICT_APRILTAG_36h11, 2, 3, 6, 6, 0.1);
  auto img = cv::imread(fs::current_path() / "../src/test/aprilgrid_6x6.png", cv::IMREAD_GRAYSCALE);

  cv::Vec3d r_vec, t_vec;
  detector.estimatePoseAprilGrid(img, camera_matrix_, dist_coeffs_, r_vec, t_vec, true);
}

TEST_F(AprilgridDetectorTest, occluded) {
  auto detector = AprilGrid(cv::aruco::DICT_APRILTAG_36h11, 2, 3, 6, 6, 0.1);
  auto img = cv::imread(fs::current_path() / "../src/test/aprilgrid_6x6_occluded.png",
                        cv::IMREAD_GRAYSCALE);

  cv::Vec3d r_vec, t_vec;
  detector.estimatePoseAprilGrid(img, camera_matrix_, dist_coeffs_, r_vec, t_vec, true);
}
