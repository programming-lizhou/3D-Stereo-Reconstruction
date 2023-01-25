//
// Created by lz on 22-10-24.
//

#ifndef STEREO_RECONSTRUCTION_DENSE_MATCHING_H
#define STEREO_RECONSTRUCTION_DENSE_MATCHING_H

#include "dataloader_mb.h"
#include <opencv2/opencv.hpp>

class DenseMatching {
public:
    DenseMatching();
    DenseMatching(Image_pair, cv::Mat, cv::Mat);
    void match(int); // mode 0: use SGBM, 1: use BM, 2: use cvCreateStereoGCState Method
    cv::Mat getDisp();
    cv::Mat getColorDisp();
private:
    Image_pair imagePair;
    cv::Mat rectified_img0;
    cv::Mat rectified_img1;
    cv::Mat disparity_map;
    cv::Mat color_disparity_map;
};



#endif //STEREO_RECONSTRUCTION_DENSE_MATCHING_H
