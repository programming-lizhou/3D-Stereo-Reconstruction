//
// Created by Li Zhou on 22-10-24.
//

#include "feature_detector.h"
#include "opencv2/xfeatures2d.hpp"


using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

Detector::Detector() = default;

Detector::Detector(Image_pair pair) {
//    this->type = tp;
    this->imagePair = pair;
    //read image to Mat
    this->img0 = imread(pair.view_path_0);
    this->img1 = imread(pair.view_path_1);
}

void Detector::setNum_features(int num) {
    this->num_features = num;
}

void Detector::setMin_hessian(int num) {
    this->min_Hessian = num;
}

void Detector::detector_SIFT() {
    //create detector
    Ptr<SIFT> detector = SIFT::create(this->num_features);
    detector->detect(this->img0, this->keypoints0);
    detector->detect(this->img1, this->keypoints1);
    //create descriptor
    Ptr<SiftDescriptorExtractor> descriptor = SiftDescriptorExtractor::create();
    descriptor->compute(this->img0, this->keypoints0, this->descriptors0);
    descriptor->compute(this->img1, this->keypoints1, this->descriptors1);

}

void Detector::detector_SURF() {
    Ptr<SURF> detector = SURF::create(this->min_Hessian);
    // detect keypoints and compute descriptors
    detector->detectAndCompute(this->img0, noArray(), this->keypoints0, this->descriptors0);
    detector->detectAndCompute(this->img1, noArray(), this->keypoints1, this->descriptors1);
}

void Detector::detector_ORB() {
    Ptr<ORB> detector = ORB::create(this->num_features);
    detector->detectAndCompute(this->img0, noArray(), this->keypoints0, this->descriptors0);
    detector->detectAndCompute(this->img1, noArray(), this->keypoints1, this->descriptors1);
    //convert the format
    if(this->descriptors0.type() != CV_32F) {
        this->descriptors0.convertTo(this->descriptors0, CV_32F);
    }

    if(this->descriptors1.type() != CV_32F) {
        this->descriptors1.convertTo(this->descriptors1, CV_32F);
    }
}

void Detector::detector_FREAK() {

}

void Detector::detector_BRISK() {


}

void Detector::detector_KAZE() {


}


vector<KeyPoint> Detector::getKeypoints0() {
    return this->keypoints0;
};

vector<KeyPoint> Detector::getKeypoints1() {
    return this->keypoints1;
};

Mat Detector::getDescriptors0() {
    return this->descriptors0;
};

Mat Detector::getDescriptors1() {
    return this->descriptors1;
};

Mat Detector::getImg0() {
    return this->img0;
};

Mat Detector::getImg1() {
    return this->img1;
};