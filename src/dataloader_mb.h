//
// Created by Li Zhou on 22-10-24.
//

#ifndef STEREO_RECONSTRUCTION_DATALOADER_MB_H
#define STEREO_RECONSTRUCTION_DATALOADER_MB_H

#include <string>

const std::string dataset_base_path = "../dataset/middlebury/";


// structure for an image pair of a dataset
struct Image_pair {
    // information from calib.txt
    float intrinsic_mtx0[3][3];
    float intrinsic_mtx1[3][3];
    float doffs;
    float baseline;
    int width;
    int height;
    int vmin;
    int vmax;
    // two view path
    std::string view_path_0;
    std::string view_path_1;
    // two disparity path
    std::string disparity_path_0;
    std::string disparity_path_1;

};

class Dataloader {
public:
    void setDataset_name(std::string);
    std::string getDataset_name() const;
    void retrievePair();
    Image_pair getPair() const;

    Dataloader();
    Dataloader(std::string);
private:
    std::string dataset_name;
    Image_pair imagePair;
};


#endif //STEREO_RECONSTRUCTION_DATALOADER_MB_H
