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





    /* example of using dtu dataloader
    Dataloader dataloader(10, 9, 1);
    dataloader.retrieveItem();
    Image_item imageItem = dataloader.getItem();

    for(int i = 0; i < imageItem.pairs.size(); i++) {
        cout << imageItem.pairs[i] << " ";
    }
    for(int i = 0; i < 4; i++) {
        for(int j = 0; j < 4; j++) {
            cout << imageItem.extrinsic_mat[i][j] << " ";
        }
    }
    cout << endl;
    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
            cout << imageItem.intrinsic_mat[i][j] << " ";
        }
    }
    cout << imageItem.depth_min << " " << imageItem.depth_interval;
    cout << imageItem.image_path << endl;
    cout << imageItem.depth_path << endl;
    cout << imageItem.depth_pfm_path << endl;
*/
    /***
    ifstream file(pair_file_path);
    string str;
//    while (getline(file, str)) {
//        cout << str << endl;
//    }
    for(int i = 0; i < 2 * 3 + 1; i++) {
        getline(file, str);

    }
    vector<int> test;
    cout << str << endl;
    istringstream iss(str);
    string word;
    int i = 0;
    while(iss >> word) {
        if(i%2) {
            test.push_back(stoi(word));
        }
        ++i;
    }
    for(int i = 0; i < test.size(); i++) {
        cout << test[i] << endl;
    }


    double extrinsic_mat[4][4];
    double intrinsic_mat[3][3];
    float depth_min;
    float depth_interval;
    string camera_path = dataset_base_path + "/Cameras/train" + "/000000" + "10" + "_cam.txt";
    //then deal with file
    ifstream camerafile(camera_path);
    string line;
    int line_no = 1;
    while(getline(camerafile, line)) {
        istringstream iss(line);
        string num;
        if(line_no >= 2 && line_no <= 5) {
            for(int j = 0; j < 4; j++) {
                iss >> num;
                extrinsic_mat[line_no - 2][j] = stod(num);
            }
        }
        if(line_no >= 8 && line_no <= 10) {
            for(int j = 0; j < 3; j++) {
                iss >> num;
                intrinsic_mat[line_no - 8][j] = stod(num);
            }
        }
        if(line_no == 12) {
            iss >> num;
            depth_min = stof(num);
            iss >> num;
            depth_interval = stof(num);
        }
        ++line_no;
    }

    for(int i = 0; i < 4; i++) {
        for(int j = 0; j < 4; j++) {
            cout << extrinsic_mat[i][j] << " ";
        }
    }
    cout << endl;
    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
            cout << intrinsic_mat[i][j] << " ";
        }
    }
    cout << endl;
    cout << depth_min << " ";
    cout <<  depth_interval;
     ***/


}