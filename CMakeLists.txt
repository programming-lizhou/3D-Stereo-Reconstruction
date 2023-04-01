cmake_minimum_required(VERSION 3.0)
project(3d_stereo_reconstruction)

set(CMAKE_CXX_STANDARD 11)

# add_executable(stereo_reconstruction src/main.cpp src/dataloader_dtu.cpp src/dataloader_dtu.h src/dataloader.cpp src/dataloader.h)

add_executable(3d_stereo_reconstruction src/main.cpp src/dataloader_mb.cpp src/dataloader_mb.h src/feature_detector.cpp src/feature_detector.h src/sparse_matching.cpp src/sparse_matching.h src/dense_matching.cpp src/dense_matching.h src/eight_point.cpp src/eight_point.h src/rectify.cpp src/rectify.h src/bundle_adjustment.cpp src/bundle_adjustment.h src/reconstruction.cpp src/reconstruction.h src/reconstruction.cpp src/reconstruction.h src/evaluation.cpp src/evaluation.h src/Eigen.h src/PFMReadWrite.h src/PFMReadWrite.cpp)

# add lib
include_directories(libs/eigen-3.4.0)

# add opencv
set(OpenCV_DIR /usr/include/opencv4/opencv2)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# add ceres
find_package(Ceres REQUIRED PATHS "/home/myname/cpp/3D-Stereo-Reconstruction/libs/ceres-solver-2.1.0")
include_directories(${CERES_INCLUDE_DIRS})

# add g2o
include_directories(libs/g2o)
SET(G2O_LIBS g2o_cli g2o_ext_freeglut_minimal g2o_simulator g2o_solver_slam2d_linear g2o_types_icp g2o_types_slam2d g2o_core g2o_interface g2o_solver_csparse g2o_solver_structure_only g2o_types_sba g2o_types_slam3d g2o_csparse_extension g2o_opengl_helper g2o_solver_dense g2o_stuff g2o_types_sclam2d g2o_parser g2o_solver_pcg g2o_types_data g2o_types_sim3 cxsparse)

target_link_libraries(3d_stereo_reconstruction ${OpenCV_LIBS} ${CERES_LIBRARIES} ${G2O_LIBS})
