#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "feature_detector.h"
#include "rectify.h"
#include "eight_point.h"
#include "dataloader_mb.h"
#include "sparse_matching.h"
#include "dense_matching.h"
using namespace cv::xfeatures2d;
using namespace std;
using namespace cv;

int main() {
    Dataloader dataloader;
    dataloader.setDataset_name("Recycle");
//    cout << dataloader.getDataset_name();
    dataloader.retrievePair();
    Image_pair imagePair = dataloader.getPair();

    int num_features = 500; //for sift and orb
    int min_hessian = 400; // for surf
    Detector detector(imagePair);
    detector.setNum_features(num_features);
//    detector.detector_SIFT();
    detector.setMin_hessian(min_hessian);
//    detector.detector_SURF();
    detector.detector_ORB();
    SparseMatching sparseMatching(1, NORM_L2);
    sparseMatching.match(detector.getDescriptors0(), detector.getDescriptors1(), 0.5);
//    sparseMatching.ransac(detector.getKeypoints0(), detector.getKeypoints1());

//    vector<DMatch> seletect_matches = sparseMatching.getGood_matches(); // here: unsorted
    vector<DMatch> seletect_matches = sparseMatching.get_sorted(); // here: sorted
    EightPointAlg eightPointAlg(imagePair.intrinsic_mtx0, imagePair.intrinsic_mtx1, detector.getKeypoints0(), detector.getKeypoints1(), seletect_matches);
    eightPointAlg.computeFMtx(1); // 0: manually, 1: opencv
    eightPointAlg.recoverRt();
    cout << eightPointAlg.getR() << endl;
    cout << eightPointAlg.getT() << endl;


    Rectify rectify = Rectify(eightPointAlg.getR(), eightPointAlg.getT(), imagePair.intrinsic_mtx0, imagePair.intrinsic_mtx1, detector.getImg0(), detector.getImg1());
    cv::imwrite("img111.png", rectify.getRectified_img0());
    cv::imwrite("img222.png", rectify.getRectified_img1());
    DenseMatching denseMatching(imagePair, rectify.getRectified_img0(), rectify.getRectified_img1());
//    DenseMatching denseMatching(imagePair, detector.getImg0(), detector.getImg1());
    denseMatching.match(0, 256, 3);
    Mat disp = denseMatching.getDisp();
    imwrite("res.png", disp);



    //here we show ground truth disp
    Mat img0 = imread(imagePair.view_path_0, 1);
    Mat img1 = imread(imagePair.view_path_1, 1);
    //DenseMatching denseMatching_gt(imagePair, detector.getImg0(), detector.getImg1());
    DenseMatching denseMatching_gt(imagePair, img0, img1);
    denseMatching_gt.match(0, 256, 3);
    Mat disp_gt = denseMatching_gt.getDisp();
    imwrite("res_gt.png", disp_gt);

    //-- Draw matches
//    cout << sparseMatching.getGood_matches().size();
/*
    Mat img_matches;
    drawMatches(detector.getImg0(), detector.getKeypoints0(), detector.getImg1(), detector.getKeypoints1(), sparseMatching.getGood_matches(), img_matches, Scalar::all(-1),
                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    //-- Show detected matches
    namedWindow("Good Matches", 0);
    resizeWindow("Good Matches", 1000, 1000);
    imshow("Good Matches", img_matches );
    waitKey();
*/
    return 0;


}