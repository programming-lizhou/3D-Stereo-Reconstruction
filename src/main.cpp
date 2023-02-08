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

bool sort_distance(DMatch dMatch1, DMatch dMatch2) {
    return dMatch1.distance < dMatch2.distance;
}

int main() {
    // init dataloader
    Dataloader dataloader;

    dataloader.setDataset_name("Piano");

    dataloader.retrievePair();
    // get image pair
    Image_pair imagePair = dataloader.getPair();

    // init detector
    Detector detector(imagePair);


//    detector.detector_SIFT();

//    detector.detector_SURF();
//    detector.detector_ORB();
//    detector.detector_FREAK();
    detector.detector_BRISK();
//    detector.detector_KAZE();

    // do sparse matching
    SparseMatching sparseMatching(0, NORM_L2);
    sparseMatching.match(detector.getDescriptors0(), detector.getDescriptors1(), detector.getKeypoints0(), detector.getKeypoints1());
    // whether to apply -------------ransac-------------, if not, comment it
    sparseMatching.ransac(detector.getKeypoints0(), detector.getKeypoints1());

    vector<DMatch> seletect_matches = sparseMatching.getGood_matches(); // here: unsorted
//    vector<DMatch> seletect_matches = sparseMatching.get_sorted(); // here: sorted


    // sparse matching
    cv::Mat RR;
    cv::Mat tt;
    cv::Mat esstenM;
    Mat k0 = Mat(3, 3, CV_32FC1, imagePair.intrinsic_mtx0);
    Mat k1 = Mat(3, 3, CV_32FC1, imagePair.intrinsic_mtx1);
    cv::recoverPose(sparseMatching.getMatched0(), sparseMatching.getMatched1(), k0, cv::noArray(), k1, cv::noArray(), esstenM, RR, tt);
    //cout << esstenM << endl;

    cout << "Five point alg:" << endl;
    cout << RR << endl;
    cout << tt << endl;

    // ground truth
    cv::Mat gt_R = (cv::Mat_<double>(3, 3) << 1,0,0,
            0,1,0,
            0,0,1);
    cv::Mat gt_T = (cv::Mat_<double>(3, 1) << -1,0,0);

    // match points
    std::vector<cv::Point2f> points0;
    std::vector<cv::Point2f> points1;
    sort(seletect_matches.begin(), seletect_matches.end(), sort_distance);
    std::vector<cv::DMatch> matches;
    for (int i=0;i<seletect_matches.size();i++){
        matches.push_back(seletect_matches[i]);
    }
    std::cout<<"number of matched points: "<<matches.size()<<std::endl;
    for(int i=0;i<matches.size();i++){
        points0.push_back(detector.getKeypoints0()[matches[i].queryIdx].pt);
        points1.push_back(detector.getKeypoints1()[matches[i].trainIdx].pt);
    }

    /*
    // get evaluation result under different settings
    cv::Mat R_temp = (cv::Mat_<double>(3, 3) << 0.9999957213846333, 0.001196905861280878, 0.002669200027366238,
    -0.001186252441654455, 0.9999913392731211, -0.003989258564518551,
    -0.002673951677111853, 0.00398607515096581, 0.9999884805273105);
    cv::Mat t_temp = (cv::Mat_<double>(3, 1) << -0.9988266035394342,
    0.0480149818016224,
    0.006322782968597357);
    std::pair<cv::Mat, cv::Mat> temp_pair = std::make_pair(R_temp, t_temp);
    std::pair<double, double> temp = evaluation.eval(temp_pair);
    std::cout<<temp.first<< " " <<temp.second<<std::endl;
*/

    // eight point algorithm
    EightPointAlg eightPointAlg(imagePair.intrinsic_mtx0, imagePair.intrinsic_mtx1, sparseMatching.getMatched0(), sparseMatching.getMatched1());
    // note, manually computing can work with Kaze, and BF
    eightPointAlg.computeFMtx(1); // 0: manually, 1: opencv
    eightPointAlg.recoverRt(1); // 0: eight, 1: five
    cout << "eightpointalg result: " << endl;
    // cout << eightPointAlg.getE() << endl;
    cout << eightPointAlg.getR() << endl;
    cout << eightPointAlg.getT() << endl;
    cv::Mat R1 = eightPointAlg.getR();
    cv::Mat t1 = eightPointAlg.getT();
    std::pair<cv::Mat, cv::Mat> eightpoint_pair = std::make_pair(R1, t1);


    // bundle adjustment to optimize pose
    BA ba(imagePair, points0, points1);
    std::pair<cv::Mat, cv::Mat> init_transformation = std::make_pair(RR, tt);
    std::pair<cv::Mat, cv::Mat> iter_transformation = ba.optimize(init_transformation, 100);
    std::cout << "BA result: " << std::endl;
    std::cout << iter_transformation.first << std::endl;
    std::cout << iter_transformation.second<< std::endl;


    // rectification
    Rectify rectify = Rectify(iter_transformation.first, iter_transformation.second, imagePair.intrinsic_mtx0, imagePair.intrinsic_mtx1, detector.getImg0(), detector.getImg1());
    cv::imwrite("img111.png", rectify.getRectified_img0());
    cv::imwrite("img222.png", rectify.getRectified_img1());

    // dense matching
    DenseMatching denseMatching(imagePair, rectify.getRectified_img0(), rectify.getRectified_img1());
//    DenseMatching denseMatching(imagePair, detector.getImg0(), detector.getImg1());
    denseMatching.match(0); // 0:sgbm, 1:bm
    Mat disp = denseMatching.getDisp();
    Mat color_disp = denseMatching.getColorDisp();
    imwrite("res.png", disp);
    imwrite("color_res.png", color_disp);

    //here we show ground truth disp
    Mat img0 = imread(imagePair.view_path_0, 1);
    Mat img1 = imread(imagePair.view_path_1, 1);
    //DenseMatching denseMatching_gt(imagePair, detector.getImg0(), detector.getImg1());
    DenseMatching denseMatching_gt(imagePair, img0, img1);
    denseMatching_gt.match(0);
    Mat disp_gt = denseMatching_gt.getDisp();
    imwrite("res_gt.png", disp_gt);
    Mat color_disp_gt = denseMatching_gt.getColorDisp();
    imwrite("color_res_gt.png", color_disp_gt);

    Mat img = loadPFM(imagePair.disparity_path_0);
    float inf = std::numeric_limits<float>::infinity();
    Mat mask = img==inf;
    img.setTo(0.0, mask);
    imwrite("disp_given.png", img);
//    cout << img << endl;
    
    Evaluation evaluation(gt_R, gt_T, imagePair);
    cout << evaluation.eval_bad(disp_gt, 1.0) << endl;
    cout << evaluation.eval_rms(img) << endl;
    
/*
    Reconstruction reconstruction(disp_gt, imagePair);
    reconstruction.calculate_depth();
    Mat dmap = reconstruction.get_dmap();
    imwrite("depth_gt.png", dmap);

    string filename = "mesh.off";
    if(reconstruction.generate_mesh(filename)) {
        cout << "cool" << endl;
    }

*/
    //-- Draw matches
    //   cout << sparseMatching.getGood_matches().size();
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