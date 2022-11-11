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
    cvtColor(img0, this->rectified_img0, COLOR_BGR2GRAY);
    cvtColor(img1, this->rectified_img1, COLOR_BGR2GRAY);
}

void DenseMatching::match(int mode, int num_disparities, int block_size) {
    if(mode == 0) {
        Ptr<StereoSGBM> sgbm = StereoSGBM::create(0, num_disparities, block_size);
        sgbm->setP1(8 * 3 * block_size * block_size);
        sgbm->setP2(32 * 3 * block_size * block_size);
        sgbm->setMode(StereoSGBM::MODE_SGBM_3WAY);
        sgbm->setSpeckleWindowSize(0);
        sgbm->setSpeckleRange(2);
        sgbm->setDisp12MaxDiff(1);
        sgbm->setUniquenessRatio(15);
        sgbm->setPreFilterCap(63);
        sgbm->compute(this->rectified_img0, this->rectified_img1, this->disparity_map);


    } else if(mode == 1) {

    }
}

Mat DenseMatching::getDisp() {
    return this->disparity_map;
}