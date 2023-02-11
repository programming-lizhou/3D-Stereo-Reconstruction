//
// Created by drla on 25.01.23.
//

#include "evaluation.h"

#include <utility>
#include "PFMReadWrite.h"
#include <cmath>

Evaluation::Evaluation(cv::Mat gt_R, cv::Mat gt_T, Image_pair ip){
    this->gt_R = std::move(gt_R);
    this->gt_T = std::move(gt_T);

    // read ground truth disparity map, using pfm helper
    cv::Mat disp = loadPFM(ip.disparity_path_0);
    // set inf values to 0.0
    float inf = std::numeric_limits<float>::infinity();
    cv::Mat mask = disp==inf;
    disp.setTo(0.0, mask);
    this->gt_disp = disp;

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
    std::pair<double, double> distance = std::make_pair(dist_R, dist_T);
    return distance;
}

// get the width of dark part on the left of the left image
int get_blocked_width(cv::Mat disp) {
    int width = INT_MAX;
    int rows = disp.rows;
    int cols = disp.cols;
    for(int i = 0; i < rows; ++i) {
        for(int j = 0; j < cols; ++j) {
            float val = disp.ptr<uchar>(i)[j];
            if(val != 0.0) {
                width = std::min(width, j);
                break;
            }
        }
    }
    return width;
}

// eval bad0.5, bad2.0, ...
double Evaluation::eval_bad(cv::Mat disp, float eval) {
    //first make sure the size is equal
    assert(disp.size() == this->gt_disp.size());

    int count = 0;
    int rows = gt_disp.rows;
    int cols = gt_disp.cols;
    int blocked_width = get_blocked_width(disp);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if(j < blocked_width) continue;
            float val = disp.ptr<uchar>(i)[j];
            float gt_val = gt_disp.at<float>(i, j);
            if(abs(val - gt_val) > eval && gt_val != 0.0) {
                ++count;
            }
        }
    }
    return count / (rows * cols * 1.0);

}


double Evaluation::eval_rms(cv::Mat disp) {
    //first make sure the size is equal
    assert(disp.size() == this->gt_disp.size());

    double result = 0.0;
    int rows = gt_disp.rows;
    int cols = gt_disp.cols;
    int blocked_width = get_blocked_width(disp);
//    std::cout << blocked_width << std::endl;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if(j < blocked_width) continue;
            float val = disp.ptr<uchar>(i)[j];
            float gt_val = gt_disp.at<float>(i, j);
            result += pow(val - gt_val, 2);
        }
    }
    return sqrt(result / (rows * (cols - blocked_width) * 1.0));

}

cv::Mat Evaluation::get_gt_disp() {
    return this->gt_disp;
}