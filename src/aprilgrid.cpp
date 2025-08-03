#include "aprilgrid.hpp"

#include <algorithm>
#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include <random>
#include <stdexcept>

AprilGrid::AprilGrid(cv::aruco::PredefinedDictionaryType dict,
                     unsigned int border_bit,
                     unsigned int separation_bits,
                     unsigned int n_rows,
                     unsigned int n_cols,
                     double marker_size)
    : dict_(dict),
      border_bits_(border_bit),
      separation_bits_(separation_bits),
      n_rows_(n_rows),
      n_cols_(n_cols),
      marker_size_(marker_size) {
  auto apriltag_data = APRILTAG_DATA_DICT.find(dict);
  if (apriltag_data == APRILTAG_DATA_DICT.end()) {
    throw std::invalid_argument("AprilGrid: Invalid dictionary type!");
  }

  // Retrieve pre-defined dictionary data
  tag_bits_ = apriltag_data->second.tag_bits;
  min_distance_ = apriltag_data->second.min_distance;
  hamming_thresh_ = apriltag_data->second.hamming_thresh;
  codes_ = apriltag_data->second.codes;

  marker_bits_ = tag_bits_ + 2 * border_bits_;
  min_cluster_pixels_ = marker_bits_ * marker_bits_;

  const int num_bits = tag_bits_ * tag_bits_;

  // Convert the integer codes into a matrix of bits (CV_8U with values 0 or
  // 1)
  tag_bit_list_ = cv::Mat(codes_.size(), num_bits, CV_8U);
  for (size_t i = 0; i < codes_.size(); ++i) {
    uint64_t code = codes_[i];
    for (int j = 0; j < num_bits; ++j) {
      tag_bit_list_.at<uchar>(i, j) = (code >> (num_bits - 1 - j)) & 1;
    }
  }
};

// Helper function: random_color
cv::Scalar AprilGrid::random_color() {
  static std::random_device rd;
  static std::mt19937 rng(rd());
  static std::uniform_int_distribution<> distribution(0, 255);
  return cv::Scalar(distribution(rng), distribution(rng), distribution(rng));
}

cv::Mat AprilGrid::poolImage(const cv::Mat &arr, int block_size, bool use_max) {
  unsigned int h = arr.rows;
  unsigned int w = arr.cols;

  unsigned int h_cropped = h - (h % block_size);
  unsigned int w_cropped = w - (w % block_size);

  cv::Rect roi(0, 0, w_cropped, h_cropped);
  cv::Mat cropped_arr = arr(roi);

  unsigned int hs = h_cropped / block_size;
  unsigned int ws = w_cropped / block_size;

  cv::Mat pooled_arr(hs, ws, arr.type());

  for (unsigned int i = 0; i < hs; ++i) {
    for (unsigned int j = 0; j < ws; ++j) {
      cv::Rect block_roi(j * block_size, i * block_size, block_size, block_size);
      cv::Mat current_block = cropped_arr(block_roi);

      double val;
      if (use_max) {
        cv::minMaxLoc(current_block, nullptr, &val);
      } else {
        cv::minMaxLoc(current_block, &val, nullptr);
      }

      // Assign the pooled value based on the matrix type
      if (arr.type() == CV_8U) {
        pooled_arr.at<uchar>(i, j) = static_cast<uchar>(val);
      } else if (arr.type() == CV_32F) {
        pooled_arr.at<float>(i, j) = static_cast<float>(val);
      } else if (arr.type() == CV_64F) {
        pooled_arr.at<double>(i, j) = static_cast<double>(val);
      } else {
        // Handle unsupported types or throw an error
        // For simplicity, we'll assume common types.
        // You might want to add a more robust error handling or type
        // conversion.
        std::cerr << "Unsupported matrix type in poolImage." << std::endl;
        pooled_arr.at<uchar>(i, j) = 0;  // Default to 0
      }
    }
  }
  return pooled_arr;
}

std::vector<AprilGrid::Detection> AprilGrid::detectTags(const cv::Mat &image_in) {
  // Check if image is valid and convert to grayscale if not already
  cv::Mat gray_img;
  if (image_in.channels() == 3) {
    cv::cvtColor(image_in, gray_img, cv::COLOR_BGR2GRAY);
  } else {
    gray_img = image_in;
  }

  // Ensure image is of an appropriate type for operations (e.g., CV_8U)
  if (gray_img.type() != CV_8U) {
    gray_img.convertTo(gray_img, CV_8U);
  }

  // step 1 resize
  int max_size = std::max(gray_img.rows, gray_img.cols);
  cv::Mat im_blur;
  cv::GaussianBlur(gray_img, im_blur, cv::Size(3, 3), 1);
  cv::Mat im_blur_resize = im_blur.clone();  // Use clone() for a distinct copy
  double new_size_ratio = 1.0;

  if (max_size > LARGE_IMAGE_THRESHOLD) {
    new_size_ratio = static_cast<double>(LARGE_IMAGE_THRESHOLD) / max_size;
    cv::resize(im_blur_resize,
               im_blur_resize,
               cv::Size(),
               new_size_ratio,
               new_size_ratio,
               cv::INTER_LINEAR);
  }

  // detect corners
  std::vector<std::vector<cv::Point>> corners_pixels = apriltagCornerThresh(im_blur_resize);

  // Convert corners from std::vector<cv::Point> to std::vector<cv::Point2f> for
  // cornerSubPix
  std::vector<std::vector<cv::Point2f>> corners_float;
  for (const auto &corner_contour : corners_pixels) {
    std::vector<cv::Point2f> float_corner;
    for (const auto &p : corner_contour) {
      float_corner.push_back(cv::Point2f(static_cast<float>(p.x), static_cast<float>(p.y)));
    }
    corners_float.push_back(float_corner);
  }

  // refine corner
  cv::Size winSize(5, 5);
  cv::Size zeroZone(-1, -1);
  cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 40, 0.001);

  // refine on small image
  if (new_size_ratio < 1.0) {
    for (auto &corner : corners_float) {
      cv::cornerSubPix(im_blur_resize, corner, winSize, zeroZone, criteria);
      for (auto &p : corner) {
        p /= new_size_ratio;
      }
    }
  }

  // refine on original image
  // Convert gray_img to float if needed for cornerSubPix, but it usually
  // handles CV_8U
  cv::Mat img_for_subpixel = image_in;  // Use original image for final refinement

  for (auto &corner : corners_float) {
    cv::cornerSubPix(img_for_subpixel, corner, winSize, zeroZone, criteria);
  }

  // Convert back to std::vector<std::vector<cv::Point>> for decodeCorner if it
  // expects it. Assuming tag_family.decodeCorner expects
  // std::vector<std::vector<cv::Point>> as in tag_family.h
  std::vector<std::vector<cv::Point2f>> final_corners_for_decode;
  for (const auto &corner_float : corners_float) {
    std::vector<cv::Point2f> int_corner;
    for (const auto &p : corner_float) {
      int_corner.push_back(cv::Point(static_cast<int>(p.x), static_cast<int>(p.y)));
    }
    final_corners_for_decode.push_back(int_corner);
  }

  std::vector<Detection> detections = decodeCorner(final_corners_for_decode, image_in);

  return detections;
}

std::vector<std::vector<cv::Point>> AprilGrid::apriltagCornerThresh(const cv::Mat &image_in) {
  // step 1. threshold the image, creating the edge image.
  cv::Mat im_copy = image_in.clone();
  cv::Mat im_thresh = thresholdImage(im_copy);

  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(im_thresh, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

  // Filter contours with less than 4 points
  std::vector<std::vector<cv::Point>> filtered_contours;
  for (const auto &c : contours) {
    if (c.size() >= 4) {
      filtered_contours.push_back(c);
    }
  }

  std::vector<std::vector<cv::Point>> corners;  // array of corner including four peak points
  for (const auto &c : filtered_contours) {
    double area = cv::contourArea(c);
    if (area > min_cluster_pixels_) {
      std::vector<cv::Point> hull;
      cv::convexHull(c, hull);
      double area_hull = cv::contourArea(hull);

      // Prevent division by zero
      if (area_hull > 0 && (area / area_hull > 0.8)) {
        std::vector<cv::Point> corner;

        // It's a tricky parameter. A common heuristic is a small percentage of arc length.
        double epsilon = cv::arcLength(hull, true) * 0.04;  // 4% of arc length,
        cv::approxPolyDP(hull, corner, epsilon, true);      // refined corner location

        if (corner.size() == 4) {
          double area_corner = cv::contourArea(corner);
          if (area_hull > 0 && area_corner > 0 && (area_corner / area_hull > 0.8) &&
              (area_hull >= area_corner)) {
            corners.push_back(corner);
          }
        }
      }
    }
  }
  return corners;
}

cv::Mat AprilGrid::thresholdImage(const cv::Mat &image_in) {
  int h = image_in.rows;
  int w = image_in.cols;

  int tile_size = 4;
  cv::Mat im_max = poolImage(image_in, tile_size, true);
  cv::Mat im_min = poolImage(image_in, tile_size, false);

  cv::Mat kernel0 = cv::Mat::ones(3, 3, CV_8U);
  cv::dilate(im_max, im_max, kernel0);
  cv::erode(im_min, im_min, kernel0);

  // This is equivalent to resizing without interpolation
  cv::Mat im_min_resized, im_max_resized;
  cv::resize(im_min, im_min_resized, cv::Size(w, h), 0, 0, cv::INTER_NEAREST);
  cv::resize(im_max, im_max_resized, cv::Size(w, h), 0, 0, cv::INTER_NEAREST);

  cv::Mat im_diff = im_max_resized - im_min_resized;
  cv::Mat im_thresh;

  // Create masks and use element-wise operations.
  // Where mask1 is true, set to 0 else if mask2 is true, set to 255, else 0.
  cv::Mat mask1;
  cv::compare(im_diff, MIN_WHITE_BLACK_DIFF, mask1, cv::CMP_LT);

  cv::Mat mask2;
  cv::Mat im_min_plus_half_diff = im_min_resized + (im_diff / 2);
  cv::compare(image_in, im_min_plus_half_diff, mask2, cv::CMP_GT);

  // Apply the masks
  im_thresh = cv::Mat::zeros(h, w, CV_8U);
  cv::Mat second_part_result = cv::Mat::zeros(h, w, CV_8U);
  second_part_result.setTo(255, mask2);

  cv::bitwise_not(mask1, mask1);
  cv::bitwise_and(mask1, second_part_result, im_thresh);

  cv::dilate(im_thresh, im_thresh, kernel0);

  // hi-res images are dilated twice
  if (std::max(im_thresh.rows, im_thresh.cols) > LARGE_IMAGE_THRESHOLD) {
    cv::dilate(im_thresh, im_thresh, kernel0);
  }

  return im_thresh;
}

void AprilGrid::decode(const cv::Mat &detected_code,
                       const std::vector<cv::Point2f> &corner,
                       std::vector<Detection> &detections) {
  cv::Mat current_code = detected_code.clone();

  for (int r = 0; r < 4; ++r) {
    cv::Mat flat_code = current_code.reshape(1, 1);  // Flatten for comparison
    int best_score_idx = -1;
    int best_score = hamming_thresh_;  // Start with the max allowed distance

    // Find the best match in the dictionary using Hamming distance
    for (int i = 0; i < tag_bit_list_.rows; ++i) {
      int score = cv::norm(flat_code, tag_bit_list_.row(i), cv::NORM_HAMMING);
      if (score < best_score) {
        best_score = score;
        best_score_idx = i;
      }
    }

    // If a good enough match is found, save it and return
    if (best_score_idx != -1) {
      std::vector<cv::Point2f> oriented_corner = corner;
      std::rotate(oriented_corner.begin(), oriented_corner.begin() + r, oriented_corner.end());
      std::reverse(oriented_corner.begin(), oriented_corner.end());
      Detection detection{best_score_idx, oriented_corner};
      detections.push_back(detection);

      return;
    }

    // If no match, rotate the code 90 degrees and try again
    cv::rotate(current_code, current_code, cv::ROTATE_90_CLOCKWISE);
  }
}

std::vector<AprilGrid::Detection> AprilGrid::decodeCorner(
    const std::vector<std::vector<cv::Point2f>> &corners,
    const cv::Mat &gray) {
  std::vector<Detection> detections;
  if (gray.empty() || gray.channels() != 1) {
    std::cerr << "Error: Input image must be a single-channel grayscale image." << std::endl;
    return detections;
  }

  for (const auto &corner : corners) {
    if (corner.size() != 4) continue;

    // Calculate total tag dimensions and define canonical corners
    float edge_pos = static_cast<float>(marker_bits_) - 0.5f;
    std::vector<cv::Point2f> tag_corners = {cv::Point2f(-0.5f, -0.5f),
                                            cv::Point2f(edge_pos, -0.5f),
                                            cv::Point2f(edge_pos, edge_pos),
                                            cv::Point2f(-0.5f, edge_pos)};

    // Find homography to warp the corner to a canonical square tag image
    cv::Mat H = cv::findHomography(corner, tag_corners);
    if (H.empty()) continue;

    // Warp the perspective to get a flat view of the tag
    cv::Mat tag_img;
    cv::warpPerspective(gray, tag_img, H, cv::Size(marker_bits_, marker_bits_));

    // Extract the inner data bits by creating a Region of Interest (ROI)
    cv::Rect data_roi(border_bits_, border_bits_, tag_bits_, tag_bits_);
    cv::Mat data_region = tag_img(data_roi);

    // Threshold the data region to get a binary code (0s and 1s)
    cv::Scalar avg_brightness = cv::mean(tag_img);
    cv::Mat detected_code;
    cv::threshold(data_region, detected_code, avg_brightness[0] + 20, 1, cv::THRESH_BINARY);
    detected_code.convertTo(detected_code, CV_8U);

    // Attempt to decode the extracted binary code
    decode(detected_code, corner, detections);
  }
  return detections;
}

/// TODO: This is not working yet
void AprilGrid::estimatePoseAprilGrid(const cv::Mat &image_in,
                                      const cv::Mat &camera_matrix,
                                      const cv::Mat &dist_coeffs,
                                      cv::Vec3d &r_vec,
                                      cv::Vec3d &t_vec) {
  auto aprilgrid_detections = detectTags(image_in);
  std::vector<cv::Point3f> predicted_corners;
  float bit_size = float(marker_size_) / float(marker_bits_);
  float tag_size = tag_bits_ * bit_size;
  for (unsigned int row = 0; row < n_rows_; row++) {
    for (unsigned int col = 0; col < n_cols_; col++) {
      float off_x = col * (marker_bits_ + separation_bits_) * bit_size;
      float off_y = row * (marker_bits_ + separation_bits_) * bit_size;
      predicted_corners.push_back(cv::Point3f(off_x + marker_size_, off_y, 0));
      predicted_corners.push_back(cv::Point3f(off_x, off_y, 0));
      predicted_corners.push_back(cv::Point3f(off_x, off_y + marker_size_, 0));
      predicted_corners.push_back(cv::Point3f(off_x + marker_size_, off_y + marker_size_, 0));
    }
  }

  std::vector<cv::Point2f> flat_corners;
  for (const auto &detection : aprilgrid_detections) {
    for (const auto &corner : detection.corners) {
      flat_corners.push_back(corner);
    }
  }

  cv::solvePnP(predicted_corners, flat_corners, camera_matrix, dist_coeffs, r_vec, t_vec);

  cv::Mat img_color;
  cv::cvtColor(image_in, img_color, cv::COLOR_GRAY2BGR);
  unsigned int i = 0;

  std::vector<cv::Point2f> projected_corners;
  cv::projectPoints(predicted_corners, r_vec, t_vec, camera_matrix, dist_coeffs, projected_corners);

  for (const auto &det : aprilgrid_detections) {
    // Calculate center of corners
    cv::Point2f center_sum(0.0f, 0.0f);
    for (int k = 0; k < 4; ++k) {
      center_sum.x += det.corners[k].x;
      center_sum.y += det.corners[k].y;
    }
    cv::Point center(static_cast<int>(std::round(center_sum.x / 4.0f)),
                     static_cast<int>(std::round(center_sum.y / 4.0f)));

    // Draw tag ID
    cv::putText(img_color,
                std::to_string(det.tag_id),
                center,
                cv::FONT_HERSHEY_SIMPLEX,
                1,
                cv::Scalar(255, 255, 0),
                2);

    // Draw corner IDs and circles
    for (int k = 0; k < 4; ++k) {
      cv::Point corner(static_cast<int>(std::round(det.corners[k].x)),
                       static_cast<int>(std::round(det.corners[k].y)));
      cv::putText(img_color,
                  std::to_string(det.tag_id * 4 + k),
                  corner,
                  cv::FONT_HERSHEY_SIMPLEX,
                  1,
                  cv::Scalar(255, 0, 255),
                  2);

      cv::line(img_color, corner, projected_corners[i], cv::Scalar(0, 255, 0), 3);

      cv::circle(img_color, projected_corners[i], 3, cv::Scalar(0, 0, 255), cv::FILLED);
      ++i;
    }
  }

  cv::imshow("Reprojection", img_color);
  cv::waitKey(0);

  cv::Mat image_copy;
  cv::cvtColor(image_in, image_copy, cv::COLOR_GRAY2RGB);
  cv::drawFrameAxes(image_copy, camera_matrix, dist_coeffs, r_vec, t_vec, 10.0);
  cv::imshow("Estimated Axis", image_copy);
  cv::waitKey(0);
}
