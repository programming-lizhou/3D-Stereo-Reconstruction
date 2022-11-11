cmake_minimum_required(VERSION 3.0)
project(stereo_reconstruction)

set(CMAKE_CXX_STANDARD 11)

# add_executable(stereo_reconstruction src/main.cpp src/dataloader_dtu.cpp src/dataloader_dtu.h src/dataloader.cpp src/dataloader.h)

add_executable(stereo_reconstruction src/main.cpp src/dataloader_mb.cpp src/dataloader_mb.h src/feature_detector.cpp src/feature_detector.h src/sparse_matching.cpp src/sparse_matching.h src/dense_matching.cpp src/dense_matching.h src/eight_point.cpp src/eight_point.h src/rectify.cpp src/rectify.h src/bundle_adjustment.cpp src/bundle_adjustment.h src/reconstruction.cpp src/reconstruction.h src/reconstruction.cpp src/reconstruction.h)

# add lib
include_directories(libs/eigen-3.4.0)

# add opencv
set(OpenCV_DIR /usr/include/opencv4/opencv2)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# add ceres
find_package(Ceres REQUIRED PATHS "/home/lz/Documents/clionprojects/3D-Stereo-Reconstruction/libs/ceres-solver-2.1.0")
include_directories(${CERES_INCLUDE_DIRS})

# add g2o


target_link_libraries(stereo_reconstruction ${OpenCV_LIBS} ${CERES_LIBRARIES})
