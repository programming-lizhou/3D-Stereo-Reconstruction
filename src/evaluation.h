//
// Created by drla on 25.01.23.
//

#ifndef STEREO_RECONSTRUCTION_EVALUATION_H
#define STEREO_RECONSTRUCTION_EVALUATION_H

// for std
#include <iostream>
// for opencv
#include <opencv2/opencv.hpp>
#include "opencv2/features2d.hpp"

class Evaluation {
public:
    Evaluation(cv::Mat, cv::Mat);
    std::pair<double, double> eval_transformation(const std::pair<cv::Mat, cv::Mat>& Transformation);
private:
    cv::Mat gt_R;
    cv::Mat gt_T;
};


#endif //STEREO_RECONSTRUCTION_EVALUATION_H
