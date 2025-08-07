# aprilgrid-cpp

A basic c++ library for performing april grid detections and pose estimations using OpenCV

<!-- TODO: Add image showing detections -->

## Build
```
mkdir build -p
cd build
cmake ../
cmake --build .
```

## Test
```
./build/aprilgrid-cpp-test
```

## Using this Library

This library aims to follow the OpenCV logical steps for fiducial board detection and pose estimation. As such, the first step is to detect all April tags from the current image.

```c++
std::vector<std::vector<cv::Point2f>> corners;
std::vector<int> ids;
detector.detectAprilTags(image, corners, ids);
```

The tag corners and IDs are used to create points representing the board in both the board and image coordinate spaces.
```c++
std::vector<cv::Point3f> obj_points;
std::vector<cv::Point2f> img_points;
detector.matchImagePoints(corners, ids, obj_points, img_points);
```

The points in these two coordinate spaces are used to find the rotation and translation vectors for the board.
```c++
cv::Vec3d r_vec, t_vec;
cv::solvePnP(obj_points, img_points, camera_matrix_, dist_coeffs_, r_vec, t_vec);
```

Finally, this library provides drawing functions to visualize the tag detection and board pose estimation.
```c++
cv::Mat image_out;
cv::cvtColor(image, image_out, cv::COLOR_GRAY2BGR);
detector.drawDetectedTags(image_out, ids, img_points);
detector.drawReprojectionErrors(image_out, ids, obj_points, img_points, r_vec, t_vec, camera_matrix_, dist_coeffs_);
cv::drawFrameAxes(image_out, camera_matrix_, dist_coeffs_, r_vec, t_vec, 5.0);
```
