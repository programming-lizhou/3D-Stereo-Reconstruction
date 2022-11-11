//
// Created by Li Zhou on 22-10-25.
//

#include "eight_point.h"
#include <Eigen/Dense>
#include <cmath>
#include <opencv2/core/eigen.hpp>

using namespace cv;
using namespace Eigen;
using namespace std;

EightPointAlg::EightPointAlg() = default;

EightPointAlg::EightPointAlg(float (*im0)[3], float (*im1)[3], vector<KeyPoint> keypoints0, vector<KeyPoint> keypoints1, vector<DMatch> good_matches) {
    Mat k_0 = Mat(3, 3, CV_32FC1, im0);
    Mat k_1 = Mat(3, 3, CV_32FC1, im1);
    k_0.convertTo(k_0, CV_64FC1);
    k_1.convertTo(k_1, CV_64FC1);
    this->k0 = k_0;
    this->k1 = k_1;
    vector<DMatch> matches;
    for(int i = 0; i < good_matches.size(); i++) {
        matches.push_back(good_matches[i]);
    }

    for(auto p = matches.begin(); p != matches.end(); p++) {
        this->points0.push_back(keypoints0.at(p->queryIdx).pt);
        this->points1.push_back(keypoints1.at(p->trainIdx).pt);
    }
}

void EightPointAlg::computeFMtx(int mode) {
    Mat FMtx;
//    cout << "test";
    if(mode == 1) {
        FMtx = findFundamentalMat(this->points0, this->points1, FM_8POINT);
    } else if(mode == 0){
        //在实现完后，经过一些测试，发现自己实现八点法的话，特征点的好坏对基础矩阵结果影响巨大，总之不够稳定。
        // mannually implement
        vector<Point2f> points80(8);
        vector<Point2f> points81(8);
        for(int i = 0; i < 8; i++) {
            points80[i] = this->points0[this->points0.size() - i - 1];
            points81[i] = this->points1[this->points1.size() - i - 1];
//            points80[i] = this->points0[i];
//            points81[i] = this->points1[i];
        }
//        cout << "test";
        //we compute the normalization matrix for pts0 and pts1.
        //reference : https://www5.cs.fau.de/fileadmin/lectures/2014s/Lecture.2014s.IMIP/exercises/4/exercise4.pdf
        //the method is called normalized 8 point alg, in order to achieve better numerical stability
        //about s, the average distance of a point p from the origin is equal to √2. So we can calculate s = √2 / dc
        float mean_x0, mean_y0, mean_x1, mean_y1;
        for(int i = 0; i < points80.size(); i++) {
            mean_x0 += points80[i].x;
            mean_y0 += points80[i].y;
            mean_x1 += points81[i].x;
            mean_y1 += points81[i].y;
        }
        mean_x0 /= 8;
        mean_y0 /= 8;
        mean_x1 /= 8;
        mean_y1 /= 8;
//        cout << mean_x0;
        float scale0, scale1, dc0_all, dc1_all;
        for(int i = 0; i < points80.size(); i++) {
            dc0_all += sqrt(pow(points80[i].x - mean_x0, 2) + pow(points80[i].y - mean_y0, 2));
            dc1_all += sqrt(pow(points81[i].x - mean_x1, 2) + pow(points81[i].y - mean_y1, 2));
        }
        //to let the avg distance to origin is sqrt(2)
        scale0 = 8 * sqrt(2) / dc0_all;
        scale1 = 8 * sqrt(2) / dc1_all;

        for(int i = 0; i < points80.size(); i++) {
            points80[i].x = scale0 * points80[i].x - scale0 * mean_x0;
            points80[i].y = scale0 * points80[i].y - scale0 * mean_y0;
            points81[i].x = scale1 * points81[i].x - scale1 * mean_x1;
            points81[i].y = scale1 * points81[i].y - scale1 * mean_y1;
        }
        // construct normalization matrix
        this->norm0 = (Mat_<float>(3, 3) << scale0, 0, -1 * scale0 * mean_x0, 0, scale0, -1 * scale0 * mean_y0, 0, 0, 1);
        this->norm1 = (Mat_<float>(3, 3) << scale1, 0, -1 * scale1 * mean_x1, 0, scale1, -1 * scale1 * mean_y1, 0, 0, 1);

        //then we construct a 8 * 9 mtx according to the course computer vision 2 slides.
        Mat mtx(8, 9, CV_32FC1);
        //for each row: a = (x1 x2 , x1 y2 , x1 z2 , y1 x2 , y1 y2 , y1 z2 , z1 x2 , z1 y2 , z1 z2 )⊤ ∈ R9
        for(int i = 0; i < 8; i++) {
            mtx.at<float>(i, 0) = points80[i].x * points81[i].x;
            mtx.at<float>(i, 1) = points80[i].x * points81[i].y;
            mtx.at<float>(i, 2) = points80[i].x * 1.0;
            mtx.at<float>(i, 3) = points80[i].y * points81[i].x;
            mtx.at<float>(i, 4) = points80[i].y * points81[i].y;
            mtx.at<float>(i, 5) = points80[i].y * 1.0;
            mtx.at<float>(i, 6) = 1.0 * points81[i].x;
            mtx.at<float>(i, 7) = 1.0 * points81[i].y;
            mtx.at<float>(i, 8) = 1.0 * 1.0;
        }
        Matrix<float,8,9> mtx_eigen;
        cv2eigen(mtx, mtx_eigen);
        //apply svd to mtx
        //JacobiSVD<MatrixXf> svd(mtx_eigen, ComputeThinU | ComputeThinV);
        JacobiSVD<MatrixXf> svd(mtx_eigen, ComputeFullU | ComputeFullV);
        // after many experiments, i find that using fullv, and i get better results
        //the result should be the last column of V
        auto matV = svd.matrixV();
        //get last column
        Matrix<float,9,1> res = matV.col(matV.cols() - 1);
        //unstack一下, then we need transpose
        Map<Matrix3f> f_before_transpose(res.data(), 3, 3);

        // with projection, according to page 14 of slide5 of cv2 course.
        Matrix3f f_after_transpose = f_before_transpose.transpose().eval();

        //project onto essential space
        JacobiSVD<MatrixXf> svd1(f_after_transpose, ComputeFullU | ComputeFullV);
        auto U = svd1.matrixU();
        auto V = svd1.matrixV();
        Matrix3f diag;
        diag << 1, 0, 0,
                0, 1, 0,
                0, 0, 0;

        Mat FMtxp;
        eigen2cv((U * diag * V.transpose()).eval(), FMtxp);
        FMtx = this->norm1.t() * FMtxp * this->norm0;
        FMtx.convertTo(FMtx, CV_64FC1);
        // end
        /* here: without projection
        Mat FMtxp;
        eigen2cv(f_before_transpose.transpose().eval(), FMtxp);

        FMtx = this->norm1.t() * FMtxp * this->norm0;
        // like in rectify.cpp, we must change the format first, or will error
        FMtx.convertTo(FMtx, CV_64FC1);
         */
        //这边还有第三种思路，就是复原normalization后，再去projection。
    }
    Mat EMtx = this->k1.t() * FMtx * this->k0;
    this->fundamentalMtx = FMtx;
    this->EssentialMtx = EMtx;

}


void EightPointAlg::recoverRt() {
    Mat RR; Mat tt;
    recoverPose(this->EssentialMtx, this->points0, this->points1, RR, tt);
    this->R = RR;
    this->t = tt;
}



Mat EightPointAlg::getR() {
    return this->R;
}


Mat EightPointAlg::getT() {
    return this->t;
}












