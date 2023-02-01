//
// Created by lz on 22-10-28.
//

#ifndef STEREO_RECONSTRUCTION_RECONSTRUCTION_H
#define STEREO_RECONSTRUCTION_RECONSTRUCTION_H


#include <opencv2/opencv.hpp>
#include "dataloader_mb.h"


class Reconstruction {
public:
    Reconstruction();
    Reconstruction(cv::Mat, Image_pair);
    void calculate_depth();
    cv::Mat get_dmap();
    bool generate_mesh(const std::string&);
private:
    cv::Mat disp_map;
    cv::Mat depth_map;
    Image_pair imagePair;
};

#endif //STEREO_RECONSTRUCTION_RECONSTRUCTION_H
