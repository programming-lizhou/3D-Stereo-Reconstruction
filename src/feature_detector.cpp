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
    this->imagePair = pair;
    //read image to Mat
    this->img0 = imread(pair.view_path_0);
    this->img1 = imread(pair.view_path_1);
}

void Detector::detector_SIFT() {
    // create SIFT detector

    // params for SIFT
    int nfeatures = 0;
    int nOctaveLayers = 4;
    float contrastThreshold = 0.04;
    int edgeThreshold = 10;
    float sigma = 1.6;

    Ptr<SIFT> detector = SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
    detector->detect(this->img0, this->keypoints0);
    detector->detect(this->img1, this->keypoints1);
    //create descriptor
    Ptr<SiftDescriptorExtractor> descriptor = SiftDescriptorExtractor::create();
    descriptor->compute(this->img0, this->keypoints0, this->descriptors0);
    descriptor->compute(this->img1, this->keypoints1, this->descriptors1);

}

void Detector::detector_SURF() {

    // params for SURF
    int hessianThreshold = 100;
    int nOctaves = 4;
    int nOctaveLayers = 4;
    bool extended = false;
    bool upright = false;


    Ptr<SURF> detector = SURF::create(hessianThreshold, nOctaves, nOctaveLayers, extended, upright);
    // detect keypoints and compute descriptors
    detector->detectAndCompute(this->img0, noArray(), this->keypoints0, this->descriptors0);
    detector->detectAndCompute(this->img1, noArray(), this->keypoints1, this->descriptors1);
}

void Detector::detector_ORB() {
    Ptr<ORB> detector = ORB::create();
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
    // Create a FREAK descriptor extractor
    /*
     * Adjust parameters:
     * 1. Whether to use orientationNormalized
     * 2. Whether to use scaleNormalized
     * 3. Adjust patternScale
     * 4. Adjust Octaves
     */
    Ptr<FastFeatureDetector> detector = FastFeatureDetector::create();
    detector->detect(this->img0, this->keypoints0);
    detector->detect(this->img1, this->keypoints1);
    // create descriptor
    Ptr<FREAK> descriptor = FREAK::create();
    descriptor->compute(this->img0, this->keypoints0, this->descriptors0);
    descriptor->compute(this->img1, this->keypoints1, this->descriptors1);

    if(descriptors0.type()!=CV_32F) {
        descriptors0.convertTo(descriptors0, CV_32F);
        }

    if(descriptors1.type()!=CV_32F) {
        descriptors1.convertTo(descriptors1, CV_32F);
    }
}

void Detector::detector_BRISK() {
    // Create a BRISK descriptor extractor
    /*
     * Adjust parameters:
     * 1. Adjust threshold
     * 2. Adjust octaves
     * 3. Adjust patternScale
     */
    // params for BRISK
    int thresh = 40;
    int octaves = 4;
    float patternScale = 1.0f;

    cv::Ptr<cv::BRISK> brisk = cv::BRISK::create(thresh, octaves, patternScale);
    // Detect keypoints and compute descriptors
    brisk->detectAndCompute(this->img0, noArray(), this->keypoints0, this->descriptors0);
    brisk->detectAndCompute(this->img1, noArray(), this->keypoints1, this->descriptors1);

    if(descriptors0.type()!=CV_32F) {
        descriptors0.convertTo(descriptors0, CV_32F);
        }

    if(descriptors1.type()!=CV_32F) {
        descriptors1.convertTo(descriptors1, CV_32F);
    }
}

void Detector::detector_KAZE() {
    // Create a KAZE descriptor extractor (Not checked yet)
    /*
     * Adjust parameters:
     * 1. Whether to use extended descriptor (128-dimensional)
     * 2. Whether to use upright descriptor
     * 3. Adjust threshold, octaves, octave layers
     * 3. Switch diffusivity (DIFF_PM_G1, DIFF_PM_G2, DIFF_WEICKERT, DIFF_CHARBONNIER)
     */
    // params for KAZE
    bool extended = false;
    bool upright = false;
    float threshold = 0.001;
    int nOctaves = 4;
    int nOctaveLayers = 4;
    cv::KAZE::DiffusivityType diffusivityType = cv::KAZE::DIFF_CHARBONNIER;

    Ptr<KAZE> kaze = KAZE::create(extended, upright, threshold, nOctaves, nOctaveLayers, diffusivityType);
    // Detect keypoints and compute descriptors
    kaze->detectAndCompute(this->img0, noArray(), this->keypoints0, this->descriptors0);
    kaze->detectAndCompute(this->img1, noArray(), this->keypoints1, this->descriptors1);
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