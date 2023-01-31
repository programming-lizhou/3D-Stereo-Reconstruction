//
// Created by lz on 22-10-28.
//

#ifndef STEREO_RECONSTRUCTION_BUNDLE_ADJUSTMENT_H
#define STEREO_RECONSTRUCTION_BUNDLE_ADJUSTMENT_H

// for std
#include <iostream>
// for opencv
#include <opencv2/opencv.hpp>
#include "opencv2/features2d.hpp"
// for g2o
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <g2o/types/slam3d/se3quat.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>

#include "dataloader_mb.h"


class BA{
public:
    BA(Image_pair, std::vector<cv::Point2f>, std::vector<cv::Point2f>);
    std::pair<cv::Mat, cv::Mat> optimize(std::pair<cv::Mat, cv::Mat>& Transformation, int iteration);

private:
    Image_pair imagePair;
    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;
    g2o::SparseOptimizer optimizer;
    Eigen::Isometry3d pose;
};

#endif //STEREO_RECONSTRUCTION_BUNDLE_ADJUSTMENT_H
