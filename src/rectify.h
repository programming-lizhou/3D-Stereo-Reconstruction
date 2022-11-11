//
// Created by lz on 22-10-28.
//

#ifndef STEREO_RECONSTRUCTION_RECTIFY_H
#define STEREO_RECONSTRUCTION_RECTIFY_H


#include <opencv2/opencv.hpp>


class Rectify {
public:
    Rectify();

    Rectify(cv::Mat R, cv::Mat t, float im0[][3], float im1[][3], cv::Mat img0, cv::Mat img1);

    cv::Mat getRectified_img0();
    cv::Mat getRectified_img1();

private:
    cv::Mat rectified_img0;
    cv::Mat rectified_img1;
};


#endif //STEREO_RECONSTRUCTION_RECTIFY_H
