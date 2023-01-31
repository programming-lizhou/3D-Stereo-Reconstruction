//
// Created by drla on 25.01.23.
//

#include "evaluation.h"

#include <utility>

Evaluation::Evaluation(cv::Mat gt_R, cv::Mat gt_T){
    this->gt_R = std::move(gt_R);
    this->gt_T = std::move(gt_T);
}

std::pair<double, double> Evaluation::eval_transformation(const std::pair<cv::Mat, cv::Mat>& Transformation){
    // set the reference matrix
    cv::Mat R = Transformation.first;
    cv::Mat T = Transformation.second;

    // initialize the result
    double dist_R = 0.0;
    double dist_T = 0.0;

    // get the L2-distance between two rotation matrices
    for(int i=0;i<3;i++){
        for(int j=0;j<3;j++){
            dist_R += std::pow(this->gt_R.at<double>(i,j) - R.at<double>(i,j), 2);
        }
    }
    dist_R /= std::pow(dist_R, 0.5);

    // get the L2-distance between two translation vectors
    for(int i=0;i<3;i++){
        dist_T += std::pow(this->gt_T.at<double>(i) - T.at<double>(i), 2);
    }
    dist_T /= std::pow(dist_T, 0.5);

    // build transformation matrix in a pair of distance (R,T)
    std::pair distance = std::make_pair(dist_R, dist_T);
    return distance;
}

