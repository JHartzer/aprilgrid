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

#ifndef APRILGRID__DETECTOR_HPP
#define APRILGRID__DETECTOR_HPP

#include <opencv2/aruco.hpp>
#include <opencv2/core.hpp>
#include <unordered_map>
#include <vector>

///
/// @class AprilGrid
/// @brief The AprilGrid class provides methods for detecting AprilGrid patterns and estimating
/// their pose.
///
/// An AprilGrid is a specific arrangement of AprilTags in a grid, which can be used for robust
/// camera calibration and pose estimation. This class handles the detection of individual tags
/// and uses them collectively to determine the pose of the entire grid.
///
class AprilGrid {
 public:
  ///
  /// @brief Constructs an AprilGrid.
  /// @param size The size of the board in x and y directions (columns and rows).
  /// @param marker_size The size of the marker in meters.
  /// @param border_bits The number of black border bits around the tag.
  /// @param separation_bits The number of bits separating adjacent tags in the grid.
  /// @param dict The predefined dictionary type for the AprilTags.
  /// @param starting_id The starting ID of the April grid.
  ///
  AprilGrid(cv::Size size,
            double marker_size,
            unsigned int border_bits,
            unsigned int separation_bits,
            int dict,
            unsigned int starting_id);

  ///
  /// @brief Detects AprilTags in an image.
  /// @param image The input image (grayscale or BGR).
  /// @param corners The corners of the decoded tags.
  /// @param ids The IDs of the decoded tags.
  ///
  void detectAprilTags(const cv::Mat &image,
                       std::vector<std::vector<cv::Point2f>> &corners,
                       std::vector<int> &ids) const;

  ///
  /// @brief Detects AprilTags in an image.
  /// @param corners The corners of the decoded tags.
  /// @param ids The IDs of the decoded tags.
  /// @param obj_points Vector of marker points in the board coordinate space.
  /// @param img_points Vector of marker points in the image coordinate space.
  ///
  void matchImagePoints(std::vector<std::vector<cv::Point2f>> &corners,
                        std::vector<int> &ids,
                        std::vector<cv::Point3f> &obj_points,
                        std::vector<cv::Point2f> &img_points) const;

  ///
  /// @brief Detects AprilTags in an image.
  /// @param image The input image (grayscale or BGR).
  /// @param ids The IDs of the decoded tags.
  /// @param img_points Vector of marker points in the image coordinate space.
  ///
  static void drawDetectedTags(cv::Mat &image,
                               std::vector<int> &ids,
                               std::vector<cv::Point2f> &img_points);

  ///
  /// @brief Detects AprilTags in an image.
  /// @param image The input image (grayscale or BGR).
  /// @param ids The IDs of the decoded tags.
  /// @param obj_points Vector of marker points in the board coordinate space.
  /// @param img_points Vector of marker points in the image coordinate space.
  /// @param r_vec Output vector corresponding to the rotation vector of the board
  /// @param t_vec Output vector corresponding to the translation vector of the board
  /// @param camera_matrix Matrix of camera coefficients
  /// @param dist_coeffs Vector of distortion coefficients
  ///
  static void drawReprojectionErrors(cv::Mat &image,
                                     std::vector<int> &ids,
                                     std::vector<cv::Point3f> &obj_points,
                                     std::vector<cv::Point2f> &img_points,
                                     cv::Vec3d r_vec,
                                     cv::Vec3d t_vec,
                                     cv::Mat camera_matrix,
                                     cv::Mat dist_coeffs);

  ///
  /// @brief Draws an image of the aprilgrid
  /// @param width The output image width
  /// @param image The output image of the aprilgrid
  ///
  void draw(unsigned int width, cv::Mat &image) const;

  /// @brief A map of the minimum distance for each pre-defined OpenCV AprilGrid dictionary
  const std::unordered_map<int, int> HAMMING_DISTANCE_MAP = {
      {cv::aruco::DICT_APRILTAG_16h5, 5},
      {cv::aruco::DICT_APRILTAG_25h9, 9},
      {cv::aruco::DICT_APRILTAG_36h10, 10},
      {cv::aruco::DICT_APRILTAG_36h11, 11}};

 private:
  ///
  /// @brief Downsamples an image by pooling.
  /// @param image The input image.
  /// @param block_size The size of the pooling window.
  /// @param use_max If true, uses max pooling. Otherwise, uses min pooling.
  /// @return The downsampled image.
  ///
  cv::Mat poolImage(const cv::Mat &image, int block_size, bool use_max) const;

  ///
  /// @brief Finds candidate corners in the image using thresholding.
  /// @param image The input grayscale image.
  /// @return A vector of contours, where each contour represents a potential tag corner cluster.
  ///
  std::vector<std::vector<cv::Point>> apriltagCornerThresh(const cv::Mat &image) const;

  ///
  /// @brief Applies an adaptive threshold to an image to binarize it.
  /// @param image The input grayscale image.
  /// @return The binarized image.
  ///
  cv::Mat thresholdImage(const cv::Mat &image) const;

  ///
  /// @brief Decodes potential AprilTags from a list of corner candidates.
  /// @param image The grayscale input image.
  /// @param candidate_corners A vector of potential tag corner sets.
  /// @param corners The corners of the decoded tags.
  /// @param ids The IDs of the decoded tags.
  /// @return A vector of successfully decoded detections.
  ///
  void decodeFromCorners(const cv::Mat &image,
                         const std::vector<std::vector<cv::Point2f>> &candidate_corners,
                         std::vector<std::vector<cv::Point2f>> &corners,
                         std::vector<int> &ids) const;

  ///
  /// @brief Decodes a single potential tag code and adds it to the list of detections if valid.
  /// @param tag_code The binary code extracted from the image for a potential tag.
  /// @param tag_corner The four corner points of the potential tag in the image.
  /// @param corners The corners of the decoded tags.
  /// @param ids The IDs of the decoded tags.
  ///
  void decodeTag(const cv::Mat &tag_code,
                 const std::vector<cv::Point2f> &tag_corner,
                 std::vector<std::vector<cv::Point2f>> &corners,
                 std::vector<int> &ids) const;

  // Member variables from constructor
  cv::aruco::Dictionary dict_;    ///< @brief The predefined AprilTags dictionary type.
  unsigned int border_bits_;      ///< @brief The number of black border bits around the tag.
  unsigned int separation_bits_;  ///< @brief The number of bits separating adjacent tags
  unsigned int n_rows_;           ///< @brief The number of rows of tags in the grid.
  unsigned int n_cols_;           ///< @brief The number of columns of tags in the grid.
  double marker_size_;            ///< @brief The size of the marker in meters
  unsigned int starting_id_;      ///< @brief The starting ID of the April grid

  // Member variables pulled from AprilGrid data dictionary
  unsigned int tag_bits_;        ///< @brief The number of data bits in the tag.
  unsigned int hamming_thresh_;  ///< @brief The Hamming distance threshold for decoding.

  // Member variables calculated in constructor
  /// @brief The total number of bits (data + border) on one side of the tag.
  unsigned int marker_bits_;
  /// @brief The minimum number of pixels for a corner cluster to be considered.
  unsigned int min_cluster_pixels_;
  /// @brief A matrix containing the binary representation of all valid tag codes.
  cv::Mat tag_bit_list_;

  // Internal configuration constants
  /// @brief Minimum difference between white and black pixels for sampling.
  const unsigned int MIN_WHITE_BLACK_DIFF{5};
  /// @brief Pixel threshold to consider an image "large" for downsampling purposes.
  const double LARGE_IMAGE_THRESHOLD{1000.0};

  // Named colors
  static const cv::Scalar CYAN;
  static const cv::Scalar MAGENTA;
  static const cv::Scalar YELLOW;
};

#endif  // APRILGRID__DETECTOR_HPP