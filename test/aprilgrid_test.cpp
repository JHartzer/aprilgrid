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

#include "aprilgrid/aprilgrid.hpp"

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
  cv::Mat dist_coeffs_ = (cv::Mat_<float>(4, 1) << 0, 0, 0, 0);
};

TEST_F(AprilgridDetectorTest, full) {
  auto april_grid = AprilGrid(cv::Size(6, 6), 0.1, 2, 3, cv::aruco::DICT_APRILTAG_36h11, 0);
  auto image_path = fs::current_path() / "../../test/assets/aprilgrid_6x6.png";
  auto image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);

  std::vector<std::vector<cv::Point2f>> corners;
  std::vector<int> ids;
  april_grid.detectAprilTags(image, corners, ids);

  std::vector<cv::Point3f> obj_points;
  std::vector<cv::Point2f> img_points;
  april_grid.matchImagePoints(corners, ids, obj_points, img_points);

  cv::Vec3d r_vec, t_vec;
  cv::solvePnP(obj_points, img_points, camera_matrix_, dist_coeffs_, r_vec, t_vec);

  cv::Mat image_out;
  cv::cvtColor(image, image_out, cv::COLOR_GRAY2BGR);
  april_grid.drawDetectedTags(image_out, ids, img_points);
  april_grid.drawReprojectionErrors(
      image_out, ids, obj_points, img_points, r_vec, t_vec, camera_matrix_, dist_coeffs_);
  cv::drawFrameAxes(image_out, camera_matrix_, dist_coeffs_, r_vec, t_vec, 0.5);

  auto out_path = fs::current_path() / "../../test/assets/aprilgrid_6x6_out.png";
  cv::imwrite(out_path, image_out);
}

TEST_F(AprilgridDetectorTest, occluded) {
  auto april_grid = AprilGrid(cv::Size(6, 6), 0.1, 2, 3, cv::aruco::DICT_APRILTAG_36h11, 0);
  auto image_path = fs::current_path() / "../../test/assets/aprilgrid_6x6_occluded.png";
  auto image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);

  std::vector<std::vector<cv::Point2f>> corners;
  std::vector<int> ids;
  april_grid.detectAprilTags(image, corners, ids);

  std::vector<cv::Point3f> obj_points;
  std::vector<cv::Point2f> img_points;
  april_grid.matchImagePoints(corners, ids, obj_points, img_points);

  cv::Vec3d r_vec, t_vec;
  cv::solvePnP(obj_points, img_points, camera_matrix_, dist_coeffs_, r_vec, t_vec);

  cv::Mat image_out;
  cv::cvtColor(image, image_out, cv::COLOR_GRAY2BGR);
  april_grid.drawDetectedTags(image_out, ids, img_points);
  april_grid.drawReprojectionErrors(
      image_out, ids, obj_points, img_points, r_vec, t_vec, camera_matrix_, dist_coeffs_);
  cv::drawFrameAxes(image_out, camera_matrix_, dist_coeffs_, r_vec, t_vec, 0.5);

  auto out_path = fs::current_path() / "../../test/assets/aprilgrid_6x6_occluded_out.png";
  cv::imwrite(out_path, image_out);
}
