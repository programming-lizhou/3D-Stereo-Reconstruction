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
#include "bundle_adjustment.h"
#include "evaluation.h"
#include "reconstruction.h"
#include "PFMReadWrite.h"
using namespace cv::xfeatures2d;
using namespace std;
using namespace cv;

enum Detectors{SIFT, SURF, ORB, FREAK, BRISK, KAZE};
enum Matchers{BF, FLANN};
enum POSECALCULATION{EIGHT_POINT, FIVE_POINT};
enum DENSEMATCHING{BM, SGBM};


const string result_dir = "../results/";

bool sort_distance(DMatch dMatch1, DMatch dMatch2) {
    return dMatch1.distance < dMatch2.distance;
}

void eval_feature_detection_and_matching(Dataloader& dataloader, Detectors det_type, Matchers matcher_type, bool ransac) {
    string filename = result_dir + "eval_feature_detection_and_matching.txt";
    ofstream ofs;
    ofs.open(filename, ios::out | ios::app);
    ofs << dataloader.getDataset_name() << ", ";

    Image_pair imagePair = dataloader.getPair();
    Detector detector(imagePair);
    switch(det_type) {
        case Detectors::SIFT:
            detector.detector_SIFT();
            ofs << "SIFT" << ", ";
            break;
        case Detectors::SURF:
            detector.detector_SURF();
            ofs << "SURF" << ", ";
            break;
        case Detectors::ORB:
            detector.detector_ORB();
            ofs << "ORB" << ", ";
            break;
        case Detectors::FREAK:
            detector.detector_FREAK();
            ofs << "FREAK" << ", ";
            break;
        case Detectors::BRISK:
            detector.detector_BRISK();
            ofs << "BRISK" << ", ";
            break;
        case Detectors::KAZE:
            detector.detector_KAZE();
            ofs << "KAZE" << ", ";
            break;
    }

    int matcher_num;
    switch(matcher_type) {
        case Matchers::BF:
            matcher_num = 0;
            ofs << "BF Matcher" << ", ";
            break;
        case Matchers::FLANN:
            matcher_num = 1;
            ofs << "FLANN Matcher" << ", ";
            break;
    }

    ofs << "Number of KeyPoints:" << detector.getKeypoints0().size() << ", ";

    SparseMatching sparseMatching(matcher_num, NORM_L2);
    sparseMatching.match(detector.getDescriptors0(), detector.getDescriptors1(), detector.getKeypoints0(), detector.getKeypoints1());

    if(ransac) {
        sparseMatching.ransac(detector.getKeypoints0(), detector.getKeypoints1());
        ofs << "With RANSAC" << ", ";
    } else {
        ofs << "Without RANSAC" << ", ";
    }
    
    ofs << "Number of Matches:" << sparseMatching.getGood_matches().size() << endl;

    ofs.close();

}

void eval_pose(Dataloader& dataloader, POSECALCULATION method, bool ransac, bool ba) {
    string filename = result_dir + "eval_pose.txt";
    ofstream ofs;
    ofs.open(filename, ios::out | ios::app);
    ofs << dataloader.getDataset_name() << ", ";

    Image_pair imagePair = dataloader.getPair();
    Detector detector(imagePair);
    detector.detector_SIFT();
    SparseMatching sparseMatching(0, NORM_L2);
    sparseMatching.match(detector.getDescriptors0(), detector.getDescriptors1(), detector.getKeypoints0(), detector.getKeypoints1());

    if(ransac) {
        sparseMatching.ransac(detector.getKeypoints0(), detector.getKeypoints1());
        ofs << "With RANSAC" << ", ";
    } else {
        ofs << "Without RANSAC" << ", ";
    }

    vector<DMatch> seletect_matches = sparseMatching.getGood_matches();
    cv::Mat gt_R = (cv::Mat_<double>(3, 3) << 1,0,0,
            0,1,0,
            0,0,1);
    cv::Mat gt_T = (cv::Mat_<double>(3, 1) << -1,0,0);

    cv::Mat R;
    cv::Mat T;
    
    EightPointAlg eightPointAlg(imagePair.intrinsic_mtx0, imagePair.intrinsic_mtx1, sparseMatching.getMatched0(), sparseMatching.getMatched1());

    if(method == POSECALCULATION::EIGHT_POINT) {
        ofs << "Eight Point Algorithm" << ", ";
        eightPointAlg.computeFMtx(1); // 0: manually, 1: opencv
        eightPointAlg.recoverRt(0); // 0: eight, 1: five
    } else if(method == POSECALCULATION::FIVE_POINT) {
        ofs << "Five Point Algorithm" << ", ";
        eightPointAlg.computeFMtx(1); // 0: manually, 1: opencv
        eightPointAlg.recoverRt(1); // 0: eight, 1: five
    }

    R = eightPointAlg.getR();
    T = eightPointAlg.getT();

    std::cout << "Without BA result: " << std::endl;
    std::cout << R << std::endl;
    std::cout << T << std::endl;

    if(ba) {
        ofs << "With BA" << ", ";
        BA ba(imagePair, sparseMatching.getMatched0(), sparseMatching.getMatched1());
        std::pair<cv::Mat, cv::Mat> init_transformation = std::make_pair(R, T);
        std::pair<cv::Mat, cv::Mat> iter_transformation = ba.optimize(init_transformation, 100);
        std::cout << "BA result: " << std::endl;
        std::cout << iter_transformation.first << std::endl;
        std::cout << iter_transformation.second<< std::endl;
        R = iter_transformation.first;
        T = iter_transformation.second;
    }
    std::pair<cv::Mat, cv::Mat> calculated_pair = std::make_pair(R, T);
    Evaluation evaluation(gt_R, gt_T, imagePair);
    pair<double, double> l2_dist = evaluation.eval_transformation(calculated_pair);
    ofs << "L2 distance of R, T : " << l2_dist.first << ", " << l2_dist.second << endl;
    
    ofs.close();
}

 void eval_DM(Dataloader& dataloader, DENSEMATCHING method, bool ba) {
    string filename = result_dir + "eval_dense_matching.txt";
    ofstream ofs;
    ofs.open(filename, ios::out | ios::app);
    ofs << dataloader.getDataset_name() << ", ";

    Image_pair imagePair = dataloader.getPair();
    Detector detector(imagePair);
    detector.detector_SIFT();
    SparseMatching sparseMatching(0, NORM_L2);
    sparseMatching.match(detector.getDescriptors0(), detector.getDescriptors1(), detector.getKeypoints0(), detector.getKeypoints1());
    sparseMatching.ransac(detector.getKeypoints0(), detector.getKeypoints1());

    vector<DMatch> seletect_matches = sparseMatching.getGood_matches();
    cv::Mat gt_R = (cv::Mat_<double>(3, 3) << 1,0,0,
            0,1,0,
            0,0,1);
    cv::Mat gt_T = (cv::Mat_<double>(3, 1) << -1,0,0);

    cv::Mat R;
    cv::Mat T;

    EightPointAlg eightPointAlg(imagePair.intrinsic_mtx0, imagePair.intrinsic_mtx1, sparseMatching.getMatched0(), sparseMatching.getMatched1());
    ofs << "Five Point Algorithm" << ", ";
    eightPointAlg.computeFMtx(1); // 0: manually, 1: opencv
    eightPointAlg.recoverRt(1); // 0: eight, 1: five
    R = eightPointAlg.getR();
    T = eightPointAlg.getT();

    std::cout << "Without BA result: " << std::endl;
    std::cout << R << std::endl;
    std::cout << T << std::endl;

    if(ba) {
        ofs << "With BA" << ", ";
        BA ba(imagePair, sparseMatching.getMatched0(), sparseMatching.getMatched1());
        std::pair<cv::Mat, cv::Mat> init_transformation = std::make_pair(R, T);
        std::pair<cv::Mat, cv::Mat> iter_transformation = ba.optimize(init_transformation, 100);
        std::cout << "BA result: " << std::endl;
        std::cout << iter_transformation.first << std::endl;
        std::cout << iter_transformation.second<< std::endl;
        R = iter_transformation.first;
        T = iter_transformation.second;
    }

    Rectify rectify = Rectify(R, T, imagePair.intrinsic_mtx0, imagePair.intrinsic_mtx1, detector.getImg0(), detector.getImg1());
    
    DenseMatching denseMatching(imagePair, rectify.getRectified_img0(), rectify.getRectified_img1());

    if(method == DENSEMATCHING::BM) {
        ofs << "BM" << ", ";
        denseMatching.match(1);
    } else if(method == DENSEMATCHING::SGBM) {
        ofs << "SGBM" << ", ";
        denseMatching.match(0);
    }

    Mat disp = denseMatching.getDisp();
    Mat color_disp = denseMatching.getColorDisp();
    imwrite("res.png", disp);
    imwrite("color_res.png", color_disp);

    Evaluation evaluation(gt_R, gt_T, imagePair);
    imwrite("disp_given.png", evaluation.get_gt_disp());

    ofs << "BAD0.5:" << evaluation.eval_bad(disp, 0.5) << ", ";
    ofs << "BAD2.0:" << evaluation.eval_bad(disp, 2.0) << ", ";
    ofs << "BAD4.0:" << evaluation.eval_bad(disp, 4.0) << ", ";
    ofs << "RMS:" << evaluation.eval_rms(disp) << endl;;

    ofs.close();
 }


// if 0, eval, else generate mesh
int eval_or_mesh = 0;

//datasets that we want to generate mesh
vector<string> dataset_names{"Playtable"};

int main() {
if(eval_or_mesh == 0) 
{
    // init dataloader
    Dataloader dataloader;
    dataloader.setDataset_name("Piano");
    dataloader.retrievePair();

    // evaluate feature detection and matching
    //    eval_feature_detection_and_matching(dataloader, Detectors::SURF, Matchers::FLANN, false);
    //    eval_feature_detection_and_matching(dataloader, Detectors::SURF, Matchers::FLANN, true);

    // evaluate pose R, t
    //    eval_pose(dataloader, POSECALCULATION::FIVE_POINT, false, true);
    //    eval_pose(dataloader, POSECALCULATION::EIGHT_POINT, true, true);

    // evaluate dense matching
    eval_DM(dataloader, DENSEMATCHING::SGBM, true);
    eval_DM(dataloader, DENSEMATCHING::BM, true);
    return 0;
}
    //vector<string> dataset_names{"Piano", "Recycle", "Playtable"};
for(auto& str : dataset_names) {
    Dataloader dataloader;
    dataloader.setDataset_name(str);
    dataloader.retrievePair();

    Image_pair imagePair = dataloader.getPair();

    Detector detector(imagePair);
    detector.detector_BRISK();

    SparseMatching sparseMatching(0, NORM_L2);
    sparseMatching.match(detector.getDescriptors0(), detector.getDescriptors1(), detector.getKeypoints0(), detector.getKeypoints1());
    sparseMatching.ransac(detector.getKeypoints0(), detector.getKeypoints1());

    EightPointAlg eightPointAlg(imagePair.intrinsic_mtx0, imagePair.intrinsic_mtx1, sparseMatching.getMatched0(), sparseMatching.getMatched1());
    eightPointAlg.computeFMtx(1); // 0: manually, 1: opencv
    eightPointAlg.recoverRt(1); // 0: eight, 1: five

    cv::Mat R = eightPointAlg.getR();
    cv::Mat T = eightPointAlg.getT();

    BA ba(imagePair, sparseMatching.getMatched0(), sparseMatching.getMatched1());
    std::pair<cv::Mat, cv::Mat> init_transformation = std::make_pair(R, T);
    std::pair<cv::Mat, cv::Mat> iter_transformation = ba.optimize(init_transformation, 100);
    std::cout << "BA result: " << std::endl;
    std::cout << iter_transformation.first << std::endl;
    std::cout << iter_transformation.second << std::endl;
    R = iter_transformation.first;
    T = iter_transformation.second;

    Rectify rectify = Rectify(R, T, imagePair.intrinsic_mtx0, imagePair.intrinsic_mtx1, detector.getImg0(), detector.getImg1());

    Mat img0 = imread(imagePair.view_path_0, 1);
    Mat img1 = imread(imagePair.view_path_1, 1);

    DenseMatching denseMatching(imagePair, img0, img1);
    denseMatching.match(0); //sgbm
    Mat disp = denseMatching.getDisp();
    
    Reconstruction reconstruction(disp, imagePair);
    reconstruction.calculate_depth();

    string filename = result_dir + str + ".off";
    if(reconstruction.generate_mesh(filename)) {
        cout << "cool!!!" << endl;
    }
}

    return 0;

}