// Copyright 2025 Jacob Hartzer
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
  auto detector = AprilGrid(cv::aruco::DICT_APRILTAG_36h11, 2, 3, 6, 6, 0.1, 0);
  auto image_path = fs::current_path() / "../src/test/assets/aprilgrid_6x6.png";
  auto image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);

  std::vector<std::vector<cv::Point2f>> corners;
  std::vector<int> ids;
  detector.detectAprilTags(image, corners, ids);

  std::vector<cv::Point3f> obj_points;
  std::vector<cv::Point2f> img_points;
  detector.matchImagePoints(corners, ids, obj_points, img_points);

  cv::Vec3d r_vec, t_vec;
  cv::solvePnP(obj_points, img_points, camera_matrix_, dist_coeffs_, r_vec, t_vec);

  cv::Mat image_out;
  cv::cvtColor(image, image_out, cv::COLOR_GRAY2BGR);
  detector.drawDetectedTags(image_out, ids, img_points);
  detector.drawReprojectionErrors(
      image_out, ids, obj_points, img_points, r_vec, t_vec, camera_matrix_, dist_coeffs_);
  cv::drawFrameAxes(image_out, camera_matrix_, dist_coeffs_, r_vec, t_vec, 5.0);

  auto out_path = fs::current_path() / "../src/test/assets/aprilgrid_6x6_out.png";
  cv::imwrite(out_path, image_out);
}

TEST_F(AprilgridDetectorTest, occluded) {
  auto detector = AprilGrid(cv::aruco::DICT_APRILTAG_36h11, 2, 3, 6, 6, 0.1, 0);
  auto image_path = fs::current_path() / "../src/test/assets/aprilgrid_6x6_occluded.png";
  auto image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);

  std::vector<std::vector<cv::Point2f>> corners;
  std::vector<int> ids;
  detector.detectAprilTags(image, corners, ids);

  std::vector<cv::Point3f> obj_points;
  std::vector<cv::Point2f> img_points;
  detector.matchImagePoints(corners, ids, obj_points, img_points);

  cv::Vec3d r_vec, t_vec;
  cv::solvePnP(obj_points, img_points, camera_matrix_, dist_coeffs_, r_vec, t_vec);

  cv::Mat image_out;
  cv::cvtColor(image, image_out, cv::COLOR_GRAY2BGR);
  detector.drawDetectedTags(image_out, ids, img_points);
  detector.drawReprojectionErrors(
      image_out, ids, obj_points, img_points, r_vec, t_vec, camera_matrix_, dist_coeffs_);
  cv::drawFrameAxes(image_out, camera_matrix_, dist_coeffs_, r_vec, t_vec, 5.0);

  auto out_path = fs::current_path() / "../src/test/assets/aprilgrid_6x6_occluded_out.png";
  cv::imwrite(out_path, image_out);
}
