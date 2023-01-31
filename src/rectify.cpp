//
// Created by lz on 22-10-28.
//

#include "rectify.h"

using namespace cv;

Rectify::Rectify() = default;

Rectify::Rectify(Mat R, Mat t, float im0[][3], float im1[][3], Mat img0, Mat img1) {
    Mat R0, R1, P0, P1, Q;
    Mat k0 = Mat(3, 3, CV_32FC1, im0);
    Mat k1 = Mat(3, 3, CV_32FC1, im1);
    k0.convertTo(k0, CV_64FC1);
    k1.convertTo(k1, CV_64FC1);

    stereoRectify(k0, noArray(), k1, noArray(), img0.size(), R, t, R0, R1, P0, P1, Q,CALIB_ZERO_DISPARITY);
    Mat map0x, map0y, map1x, map1y;
    Mat rimg0, rimg1;

    initUndistortRectifyMap(k0, noArray(), R0, k0, img0.size(), CV_32FC1, map0x, map0y);
    initUndistortRectifyMap(k1, noArray(), R1, k1, img1.size(), CV_32FC1, map1x, map1y);

    remap(img0, rimg0, map0x, map0y, INTER_LINEAR, BORDER_CONSTANT, Scalar());
    remap(img1, rimg1, map1x, map1y, INTER_LINEAR, BORDER_CONSTANT, Scalar());
    this->rectified_img0 = rimg0;
    this->rectified_img1 = rimg1;
}



Mat Rectify::getRectified_img0() {
    return this->rectified_img0;
}

Mat Rectify::getRectified_img1() {
    return this->rectified_img1;
}