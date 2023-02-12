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
#include "dataloader_mb.h"

class Evaluation {
public:
    Evaluation(cv::Mat, cv::Mat, Image_pair ip);
    std::pair<double, double> eval_transformation(const std::pair<cv::Mat, cv::Mat>& Transformation);
    double eval_bad(cv::Mat disp, float eval);
    double eval_rms(cv::Mat disp);
    cv::Mat get_gt_disp();
private:
    cv::Mat gt_R;
    cv::Mat gt_T;
    cv::Mat gt_disp;
    cv::Mat gt_disp_origin;
};


#endif //STEREO_RECONSTRUCTION_EVALUATION_H
