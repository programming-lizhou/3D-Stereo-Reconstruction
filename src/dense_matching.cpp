//
// Created by lz on 22-10-24.
//

#include "dense_matching.h"
using namespace cv;


DenseMatching::DenseMatching() = default;

DenseMatching::DenseMatching(Image_pair ip, Mat img0, Mat img1) {
    this->imagePair = ip;
//    this->rectified_img0 = img0;
//    this->rectified_img1 = img1;
    this->rectified_img0.create(img0.size(), CV_8UC1);
    this->rectified_img1.create(img1.size(), CV_8UC1);
    cvtColor(img0, this->rectified_img0, COLOR_BGR2GRAY);
    cvtColor(img1, this->rectified_img1, COLOR_BGR2GRAY);
}

void DenseMatching::match(int mode) {
    if(mode == 0) { // 0:sgbm, 1:bm
        Ptr<StereoSGBM> sgbm = StereoSGBM::create();

        int blockSize = 6;
        int numDisparities = 256;
        int minDisparity = 0;
        int uniquenessRatio = 10;
        int speckleRange = 32;
        int speckleWindowSize = 100;
        int preFilterCap = 58;
        int disp12MaxDiff = 1;
        int p1 = 8 * 1 * blockSize * blockSize;
        int p2 = 64 * 1 * blockSize * blockSize;


        sgbm->setNumDisparities(numDisparities);
        sgbm->setMinDisparity(minDisparity);
        sgbm->setBlockSize(blockSize);
        sgbm->setP1(p1);
        sgbm->setP2(p2);
        sgbm->setMode(StereoSGBM::MODE_SGBM);
        sgbm->setSpeckleWindowSize(speckleWindowSize);
        sgbm->setSpeckleRange(speckleRange);
        sgbm->setDisp12MaxDiff(disp12MaxDiff);
        sgbm->setUniquenessRatio(uniquenessRatio);
        sgbm->setPreFilterCap(preFilterCap);


        sgbm->compute(this->rectified_img0, this->rectified_img1, this->disparity_map);
        this->disparity_map.convertTo(this->disparity_map, CV_8U, 255/(numDisparities*16.));
        applyColorMap(this->disparity_map, this->color_disparity_map, cv::COLORMAP_JET);
    } else if(mode == 1) {
        int numDisparities = 256;
        int blockSize = 25;
        int preFilterCap = 58;
        int uniquenessRatio = 10;

        Ptr<StereoBM> bm = StereoBM::create(numDisparities, blockSize);

        bm->setPreFilterCap(preFilterCap);
        bm->setUniquenessRatio(uniquenessRatio);

        bm->compute(this->rectified_img0, this->rectified_img1, this->disparity_map);
        this->disparity_map.convertTo(this->disparity_map, CV_8U, 255/(numDisparities*16.));
        applyColorMap(this->disparity_map, this->color_disparity_map, cv::COLORMAP_JET);
    }
}

Mat DenseMatching::getDisp() {
    return this->disparity_map;
}

Mat DenseMatching::getColorDisp() {
    return this->color_disparity_map;
}