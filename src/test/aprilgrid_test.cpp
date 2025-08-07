

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
  auto image_path = fs::current_path() / "src/test/assets/aprilgrid_6x6.png";
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

  auto out_path = fs::current_path() / "src/test/assets/aprilgrid_6x6_out.png";
  cv::imwrite(out_path, image_out);
}

TEST_F(AprilgridDetectorTest, occluded) {
  auto detector = AprilGrid(cv::aruco::DICT_APRILTAG_36h11, 2, 3, 6, 6, 0.1);
  auto image_path = fs::current_path() / "src/test/assets/aprilgrid_6x6_occluded.png";
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

  auto out_path = fs::current_path() / "src/test/assets/aprilgrid_6x6_occluded_out.png";
  cv::imwrite(out_path, image_out);
}
