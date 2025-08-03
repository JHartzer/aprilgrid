

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

TEST(test_detector, aprilgrid_b2) {
  auto detector = AprilGrid(cv::aruco::DICT_APRILTAG_36h11, 2, 3, 6, 6, 0.1);
  auto img = cv::imread(fs::current_path() / "../src/test/aprilgrid_6x6.png", cv::IMREAD_GRAYSCALE);

  cv::Vec3d r_vec;
  cv::Vec3d t_vec;
  cv::Mat camera_matrix = cv::Mat::eye(3, 3, CV_64F);
  camera_matrix.at<double>(0, 0) = 500.0;
  camera_matrix.at<double>(1, 1) = 500.0;
  camera_matrix.at<double>(0, 2) = 512.0;
  camera_matrix.at<double>(1, 2) = 512.0;
  cv::Mat dist_coeffs{0, 0, 0, 0};

  detector.estimatePoseAprilGrid(img, camera_matrix, dist_coeffs, r_vec, t_vec);

  std::cout << r_vec << std::endl;
  std::cout << t_vec << std::endl;
}
