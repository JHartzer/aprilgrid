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

#include <opencv2/opencv.hpp>

const cv::Scalar AprilGrid::CYAN{255, 255, 0};
const cv::Scalar AprilGrid::MAGENTA{255, 0, 255};
const cv::Scalar AprilGrid::YELLOW{0, 255, 255};

AprilGrid::AprilGrid(cv::aruco::PredefinedDictionaryType dict,
                     unsigned int border_bit,
                     unsigned int separation_bits,
                     unsigned int n_rows,
                     unsigned int n_cols,
                     double marker_size,
                     unsigned int starting_id)
    : dict_(dict),
      border_bits_(border_bit),
      separation_bits_(separation_bits),
      n_rows_(n_rows),
      n_cols_(n_cols),
      marker_size_(marker_size),
      starting_id_(starting_id) {
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

  // Convert the integer codes into a matrix of bits (CV_8U with values 0 or 1)
  tag_bit_list_ = cv::Mat(codes_.size(), num_bits, CV_8U);
  for (size_t i = 0; i < codes_.size(); ++i) {
    uint64_t code = codes_[i];
    for (int j = 0; j < num_bits; ++j) {
      tag_bit_list_.at<uchar>(i, j) = (code >> (num_bits - 1 - j)) & 1;
    }
  }
};

cv::Mat AprilGrid::poolImage(const cv::Mat &image, int block_size, bool use_max) {
  unsigned int h = image.rows;
  unsigned int w = image.cols;

  unsigned int h_cropped = h - (h % block_size);
  unsigned int w_cropped = w - (w % block_size);

  unsigned int hs = h_cropped / block_size;
  unsigned int ws = w_cropped / block_size;

  cv::Mat pooled_img(hs, ws, image.type());

  // Process blocks for CV_8U (most common case)
  if (image.type() == CV_8U) {
    const uchar *data = image.ptr<uchar>();
    uchar *pooled_data = pooled_img.ptr<uchar>();

    for (unsigned int i = 0; i < hs; ++i) {
      for (unsigned int j = 0; j < ws; ++j) {
        uchar val = use_max ? 0 : 255;

        // Process block without creating sub-matrix
        for (int dy = 0; dy < block_size; ++dy) {
          const uchar *row = data + (i * block_size + dy) * w + j * block_size;
          for (int dx = 0; dx < block_size; ++dx) {
            uchar pixel = row[dx];
            if (use_max) {
              val = std::max(val, pixel);
            } else {
              val = std::min(val, pixel);
            }
          }
        }
        pooled_data[i * ws + j] = val;
      }
    }
  } else {
    // Fallback for other types
    cv::Rect roi(0, 0, w_cropped, h_cropped);
    cv::Mat cropped_arr = image(roi);

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

        if (image.type() == CV_32F) {
          pooled_img.at<float>(i, j) = static_cast<float>(val);
        } else if (image.type() == CV_64F) {
          pooled_img.at<double>(i, j) = static_cast<double>(val);
        }
      }
    }
  }
  return pooled_img;
}

void AprilGrid::detectAprilTags(const cv::Mat &image,
                                std::vector<std::vector<cv::Point2f>> &corners,
                                std::vector<int> &ids) {
  // Check if image is valid and convert to grayscale if not already
  cv::Mat gray_img;
  if (image.channels() == 3) {
    cv::cvtColor(image, gray_img, cv::COLOR_BGR2GRAY);
  } else {
    gray_img = image;
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
  // Convert gray_img to float if needed for cornerSubPix, but it usually handles CV_8U
  cv::Mat img_for_subpixel = image;  // Use original image for final refinement

  for (auto &corner : corners_float) {
    cv::cornerSubPix(img_for_subpixel, corner, winSize, zeroZone, criteria);
  }

  // Convert back to std::vector<std::vector<cv::Point>> for decodeFromCorners if it expects it
  std::vector<std::vector<cv::Point2f>> final_corners_for_decode;
  for (const auto &corner_float : corners_float) {
    std::vector<cv::Point2f> int_corner;
    for (const auto &p : corner_float) {
      int_corner.push_back(cv::Point(static_cast<int>(p.x), static_cast<int>(p.y)));
    }
    final_corners_for_decode.push_back(int_corner);
  }

  decodeFromCorners(image, final_corners_for_decode, corners, ids);
}

std::vector<std::vector<cv::Point>> AprilGrid::apriltagCornerThresh(const cv::Mat &image) {
  // step 1. threshold the image, creating the edge image.
  cv::Mat im_copy = image.clone();
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

cv::Mat AprilGrid::thresholdImage(const cv::Mat &image) {
  int h = image.rows;
  int w = image.cols;

  int tile_size = 4;
  cv::Mat im_max = poolImage(image, tile_size, true);
  cv::Mat im_min = poolImage(image, tile_size, false);

  cv::Mat kernel0 = cv::Mat::ones(3, 3, CV_8U);
  cv::dilate(im_max, im_max, kernel0);
  cv::erode(im_min, im_min, kernel0);

  // This is equivalent to resizing without interpolation
  cv::Mat im_min_resized, im_max_resized;
  cv::resize(im_min, im_min_resized, cv::Size(w, h), 0, 0, cv::INTER_NEAREST);
  cv::resize(im_max, im_max_resized, cv::Size(w, h), 0, 0, cv::INTER_NEAREST);

  cv::Mat im_diff = im_max_resized - im_min_resized;

  // Create masks and use element-wise operations.
  // Where mask1 is true, set to 0 else if mask2 is true, set to 255, else 0.
  cv::Mat mask1;
  cv::compare(im_diff, MIN_WHITE_BLACK_DIFF, mask1, cv::CMP_LT);

  cv::Mat mask2;
  cv::Mat im_min_plus_half_diff = im_min_resized + (im_diff / 2);
  cv::compare(image, im_min_plus_half_diff, mask2, cv::CMP_GT);

  // Apply the masks
  cv::Mat im_thresh = cv::Mat::zeros(h, w, CV_8U);
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

void AprilGrid::decodeTag(const cv::Mat &tag_code,
                          const std::vector<cv::Point2f> &tag_corner,
                          std::vector<std::vector<cv::Point2f>> &corners,
                          std::vector<int> &ids) {
  const int code_size = tag_code.rows * tag_code.cols;

  // Pre-allocate rotation buffers to avoid repeated memory allocation
  static thread_local std::vector<uchar> current_bits(64);  // Max expected size
  static thread_local std::vector<uchar> rotated_bits(64);

  current_bits.resize(code_size);
  rotated_bits.resize(code_size);

  // Copy initial code to buffer
  const uchar *code_data = tag_code.ptr<uchar>();
  std::memcpy(current_bits.data(), code_data, code_size);

  const int rows = tag_code.rows;
  const int cols = tag_code.cols;

  for (int r = 0; r < 4; ++r) {
    int best_score_idx = -1;
    int best_score = hamming_thresh_;

    // Optimized Hamming distance calculation
    const uchar *tag_data = tag_bit_list_.ptr<uchar>();
    for (int i = 0; i < tag_bit_list_.rows; ++i) {
      int score = 0;
      const uchar *tag_row = tag_data + i * code_size;

      // Manual Hamming distance for better performance
      for (int j = 0; j < code_size; ++j) {
        score += (current_bits[j] != tag_row[j]);
        if (score >= best_score) break;  // Early termination
      }

      if (score < best_score) {
        best_score = score;
        best_score_idx = i;
      }
    }

    if (best_score_idx != -1) {
      std::vector<cv::Point2f> oriented_corner = tag_corner;
      std::rotate(oriented_corner.begin(), oriented_corner.begin() + r, oriented_corner.end());
      std::reverse(oriented_corner.begin(), oriented_corner.end());
      corners.push_back(oriented_corner);
      ids.push_back(best_score_idx);
      return;
    }

    // Optimized 90-degree rotation without cv::rotate
    if (r < 3) {  // No need to rotate on last iteration
      for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
          rotated_bits[j * rows + (rows - 1 - i)] = current_bits[i * cols + j];
        }
      }
      current_bits.swap(rotated_bits);
    }
  }
}

void AprilGrid::decodeFromCorners(const cv::Mat &image,
                                  const std::vector<std::vector<cv::Point2f>> &candidate_corners,
                                  std::vector<std::vector<cv::Point2f>> &corners,
                                  std::vector<int> &ids) {
  if (image.empty() || image.channels() != 1) {
    std::cerr << "Error: Input image must be a single-channel grayscale image." << std::endl;
    return;
  }

  for (const auto &tag_corners : candidate_corners) {
    if (tag_corners.size() != 4) continue;

    // Calculate total tag dimensions and define canonical corners
    float edge_pos = static_cast<float>(marker_bits_) - 0.5f;
    std::vector<cv::Point2f> corners_est = {cv::Point2f(-0.5f, -0.5f),
                                            cv::Point2f(edge_pos, -0.5f),
                                            cv::Point2f(edge_pos, edge_pos),
                                            cv::Point2f(-0.5f, edge_pos)};

    // Find homography to warp the tag_corners to a canonical square tag image
    cv::Mat H = cv::findHomography(tag_corners, corners_est);
    if (H.empty()) continue;

    // Warp the perspective to get a flat view of the tag
    cv::Mat tag_img;
    cv::warpPerspective(image, tag_img, H, cv::Size(marker_bits_, marker_bits_));

    // Extract the inner data bits by creating a Region of Interest (ROI)
    cv::Rect data_roi(border_bits_, border_bits_, tag_bits_, tag_bits_);
    cv::Mat data_region = tag_img(data_roi);

    // Threshold the data region to get a binary code (0s and 1s)
    cv::Scalar avg_brightness = cv::mean(tag_img);
    cv::Mat tag_code;
    cv::threshold(data_region, tag_code, avg_brightness[0] + 20, 1, cv::THRESH_BINARY);
    tag_code.convertTo(tag_code, CV_8U);

    // Attempt to decode the extracted binary code
    decodeTag(tag_code, tag_corners, corners, ids);
  }
}

void AprilGrid::matchImagePoints(std::vector<std::vector<cv::Point2f>> &corners,
                                 std::vector<int> &ids,
                                 std::vector<cv::Point3f> &obj_points,
                                 std::vector<cv::Point2f> &img_points) {
  if (ids.empty()) {
    return;  // Early exit if no detections
  }

  // Flatten detected corners more efficiently
  img_points.reserve(ids.size() * 4);

  for (const auto &tag_corners : corners) {
    img_points.insert(img_points.end(), tag_corners.begin(), tag_corners.end());
  }

  // Pre-calculate constants once
  const float bit_size = float(marker_size_) / float(marker_bits_);
  const float grid_size = (marker_bits_ + separation_bits_) * bit_size;
  const float grid_width = (marker_bits_ * n_cols_ + (n_cols_ - 1) * separation_bits_) * bit_size;

  // Generate predicted 3D points
  for (const auto &tag_id : ids) {
    int row = (tag_id - starting_id_) / n_cols_;
    int col = (tag_id - starting_id_) % n_cols_;
    const float off_y = row * grid_size;
    const float off_x = col * grid_size;
    const float x1 = grid_width - (off_x + marker_size_);
    const float x2 = grid_width - off_x;
    const float y1 = off_y;
    const float y2 = off_y + marker_size_;

    obj_points.emplace_back(x2, y1, 0.0f);
    obj_points.emplace_back(x1, y1, 0.0f);
    obj_points.emplace_back(x1, y2, 0.0f);
    obj_points.emplace_back(x2, y2, 0.0f);
  }
}

void AprilGrid::drawDetectedTags(cv::Mat &image,
                                 std::vector<int> &ids,
                                 std::vector<cv::Point2f> &img_points) {
  unsigned int i = 0;
  for (unsigned int i = 0; i < ids.size(); ++i) {
    cv::Point center(0.0, 0.0);

    for (int j = 0; j < 4; ++j) {
      int k = 4 * i + j;
      auto corner_id = std::to_string(4 * ids[i] + j);
      cv::putText(image, corner_id, img_points[k], cv::FONT_HERSHEY_SIMPLEX, 0.5, MAGENTA, 1);
      cv::circle(image, img_points[k], 2, MAGENTA, cv::FILLED);
      center.x += img_points[k].x;
      center.y += img_points[k].y;
    }

    center.x *= 0.25;
    center.y *= 0.25;

    cv::putText(image, std::to_string(ids[i]), center, cv::FONT_HERSHEY_SIMPLEX, 1.0, CYAN, 2);
  }
}

void AprilGrid::drawReprojectionErrors(cv::Mat &image,
                                       std::vector<int> &ids,
                                       std::vector<cv::Point3f> &obj_points,
                                       std::vector<cv::Point2f> &img_points,
                                       cv::Vec3d r_vec,
                                       cv::Vec3d t_vec,
                                       cv::Mat camera_matrix,
                                       cv::Mat dist_coeffs) {
  std::vector<cv::Point2f> projected_corners;
  cv::projectPoints(obj_points, r_vec, t_vec, camera_matrix, dist_coeffs, projected_corners);
  for (unsigned int i = 0; i < ids.size(); ++i) {
    cv::line(image, img_points[i], projected_corners[i], YELLOW, 2);
  }
}