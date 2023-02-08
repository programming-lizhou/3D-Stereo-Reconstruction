//
// Created by drla on 01.02.23.
//

#include<iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace cv;
int main( int argc, char** argv )
{

    Mat rgb1 = imread( "/home/drla/Documents/study/3d_scanning/project/3D-Stereo-Reconstruction/build/img111.png" );
    Mat rgb2 = imread( "/home/drla/Documents/study/3d_scanning/project/3D-Stereo-Reconstruction/build/img222.png" );

//    Ptr<FeatureDetector> detector;
//    Ptr<DescriptorExtractor> descriptor;
//    detector = FeatureDetector::create("ORB");
//    descriptor = DescriptorExtractor::create("ORB");

    // used in OpenCV3
    int nfeatures = 0;
    int nOctaveLayers = 4;
    float contrastThreshold = 0.04;
    int edgeThreshold = 10;
    float sigma = 1.6;

    //Ptr<SIFT> detector = SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
    //Ptr<SiftDescriptorExtractor> descriptor = SiftDescriptorExtractor::create();
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    // use this if you are in OpenCV2
    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );

    vector< KeyPoint > kp1, kp2;
    detector->detect( rgb1, kp1 );
    detector->detect( rgb2, kp2 );

    // 计算描述子
    Mat desp1, desp2;
    descriptor->compute( rgb1, kp1, desp1 );
    descriptor->compute( rgb2, kp2, desp2 );

    // 匹配描述子
    vector< DMatch > matches;
    BFMatcher matcher;

    matcher.match( desp1, desp2, matches );
    cout<<"Find total "<<matches.size()<<" matches."<<endl;

    // 筛选匹配对
    vector< DMatch > goodMatches;
    double minDis = 9999;
    for ( size_t i=0; i<matches.size(); i++ )
    {
        if ( matches[i].distance < minDis )
            minDis = matches[i].distance;
    }

    for ( size_t i=0; i<matches.size(); i++ )
    {
        if (matches[i].distance < 10*minDis)
            goodMatches.push_back( matches[i] );
    }


    vector< Point2f > pts1, pts2;
    for (size_t i=0; i<goodMatches.size(); i++)
    {
        pts1.push_back(kp1[goodMatches[i].queryIdx].pt);
        pts2.push_back(kp2[goodMatches[i].trainIdx].pt);
    }

    // 请先计算基础矩阵并据此绘制出前10个匹配点对应的对极线，可以调用opencv函数

    //首先根据对应点计算出两视图的基础矩阵，基础矩阵包含了两个相机的外参数关系
    Mat fundamental_matrix=findFundamentalMat(pts1,pts2,FM_8POINT);
    //计算对应点的外极线epilines是一个三元组(a,b,c)，表示点在另一视图中对应的外极线ax+by+c=0;
    vector<cv::Vec<float, 3>> epilines1,epilines2;
    computeCorrespondEpilines(pts1,1,fundamental_matrix,epilines1);
    computeCorrespondEpilines(pts2,2,fundamental_matrix,epilines2);
    cv::RNG &rng = theRNG();
    for (int i = 0; i < 5; ++i) {
        //随机产生颜色
        Scalar color = Scalar(rng(255), rng(255), rng(255));
        circle(rgb1, pts1[i], 5, color, 3);
        //绘制外极线的时候，选择两个点，一个是x=0处的点，一个是x为图片宽度处
        line(rgb1, cv::Point(0, -epilines2[i][2] / epilines2[i][1]),Point(rgb1.cols, -(epilines2[i][2] + epilines2[i][0] * rgb1.cols) / epilines2[i][1]), color);

        circle(rgb2, pts2[i], 5, color, 3);
        line(rgb2, cv::Point(0, -epilines1[i][2] / epilines1[i][1]),Point(rgb2.cols, -(epilines1[i][2] + epilines1[i][0] * rgb2.cols) / epilines1[i][1]), color);
    }

    imwrite("epiline1.jpg",rgb2);
    imwrite("epiline2.jpg",rgb1);
    waitKey(0);
    return 0;
}
