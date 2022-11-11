//
// Created by Li Zhou on 22-10-24.
//

#ifndef STEREO_RECONSTRUCTION_FEATURE_DETECTOR_H
#define STEREO_RECONSTRUCTION_FEATURE_DETECTOR_H

#include <vector>
#include <opencv2/opencv.hpp>
#include "dataloader_mb.h"

//enum Detector_types {SIFT, SURF, ORB, FREAK};


class Detector {
public:
    Detector();
    Detector(Image_pair);


//    void detect();
    void detector_SIFT();
    void detector_SURF();
    void detector_ORB();
    void detector_FREAK();
    void detector_BRISK();
    void detector_KAZE();


    void setNum_features(int);
    void setMin_hessian(int);

    std::vector<cv::KeyPoint> getKeypoints0();
    std::vector<cv::KeyPoint> getKeypoints1();

    cv::Mat getDescriptors0();
    cv::Mat getDescriptors1();
    cv::Mat getImg0();
    cv::Mat getImg1();
private:
//    int type;

    std::vector<cv::KeyPoint> keypoints0;
    std::vector<cv::KeyPoint> keypoints1;
    cv::Mat descriptors0;
    cv::Mat descriptors1;
    cv::Mat img0;
    cv::Mat img1;

    Image_pair imagePair;

    int num_features; // for SIFT, ORB
    int min_Hessian; // for SURF

};




#endif //STEREO_RECONSTRUCTION_FEATURE_DETECTOR_H
