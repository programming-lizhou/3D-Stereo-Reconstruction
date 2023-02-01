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
    // create empty depth map
    cv::Mat dmap(disp_map.size(), CV_32FC1);

    int rows = dmap.rows;
    int cols = dmap.cols;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float d = disp_map.ptr<uchar>(i)[j];
            if (d == 0) continue;
            dmap.ptr<float>(i)[j] = this->imagePair.baseline * imagePair.intrinsic_mtx0[0][0] / (d+this->imagePair.doffs);
        }
    }
    cv::Mat depth_norm;
    cv::normalize(dmap, depth_norm, 0.0, 1.0, cv::NORM_MINMAX);
    depth_norm *= 255.0;
    this->depth_map = depth_norm;
}

cv::Mat Reconstruction::get_dmap() {
    return this->depth_map;
}