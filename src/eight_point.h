//
// Created by Li Zhou on 22-10-25.
//

#ifndef STEREO_RECONSTRUCTION_EIGHT_POINT_H
#define STEREO_RECONSTRUCTION_EIGHT_POINT_H

#include <vector>
#include <opencv2/opencv.hpp>
#include "opencv2/features2d.hpp"

class EightPointAlg {
public:
    EightPointAlg();

    EightPointAlg(float im0[][3], float im1[][3], std::vector<cv::KeyPoint>, std::vector<cv::KeyPoint>, std::vector<cv::DMatch>);

    void computeFMtx(int mode);

    void recoverRt();

    cv::Mat getR();

    cv::Mat getT();

private:
    //int mode; // if 0, use the version implemented by myself, if 1, use the method from opencv.
    std::vector<cv::Point2f> points0;
    std::vector<cv::Point2f> points1;
    cv::Mat norm0;
    cv::Mat norm1;
    cv::Mat fundamentalMtx;
    cv::Mat EssentialMtx;
    cv::Mat R;
    cv::Mat t;
    cv::Mat k0;
    cv::Mat k1;
};


#endif //STEREO_RECONSTRUCTION_EIGHT_POINT_H
