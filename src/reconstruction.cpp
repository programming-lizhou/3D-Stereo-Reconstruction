//
// Created by lz on 22-10-28.
//

#include "reconstruction.h"

#include <utility>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

Reconstruction::Reconstruction() = default;

Reconstruction::Reconstruction(cv::Mat disp, Image_pair ip) {
    this->disp_map = std::move(disp);
    this->imagePair = std::move(ip);
}


void Reconstruction::calculate_depth() {
    /*
    cv::Mat dmap(disp_map.rows, disp_map.cols, CV_32FC1);
    std::cout << "test0" << std::endl;
    int nrow = dmap.rows;
    int ncol = dmap.cols;
    for(int i = 0; i < nrow; ++i) {
        for(int j = 0; j < ncol; ++j) {
            float d = disp_map.at<float>(i, j);
            dmap.at<float>(i, j) = imagePair.intrinsic_mtx0[0][0] * imagePair.baseline / (d * 0.0625 + imagePair.doffs);
        }
    }
    this->depth_map = dmap;
    std::cout << "test1" << std::endl;
     */
    cv::Mat dmap(disp_map.size(), CV_32FC1);

    int rows = dmap.rows;
    int cols = dmap.cols;

    for (int x = 0; x < rows; x++)
    {
        for (int y = 0; y < cols; y++)
        {
            float d = disp_map.ptr<uchar>(x)[y];
            if (d == 0)
                continue;
            dmap.ptr<float>(x)[y] = this->imagePair.baseline * imagePair.intrinsic_mtx0[0][0] / (d+this->imagePair.doffs);
            //dmap.ptr<short>(y, x) = depth_val;

        }
    }
    cv::Mat depth_norm;
    cv::normalize(dmap, depth_norm, 0.0, 1.0, cv::NORM_MINMAX);
    depth_norm *= 255.0;
    //std::cout<<depth_norm<<std::endl;
    this->depth_map = depth_norm;
}

cv::Mat Reconstruction::get_dmap() {
    return this->depth_map;
}