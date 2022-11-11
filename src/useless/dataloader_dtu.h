//
// Created by Li Zhou on 22-10-21.
//

#ifndef STEREO_RECONSTRUCTION_DATALOADER_DTU_H
#define STEREO_RECONSTRUCTION_DATALOADER_DTU_H

#include <string>
#include <vector>
//#include <opencv2/opencv.hpp>
//#include <opencv2/core.hpp>

//path for dataset base
const std::string dataset_base_path = "../dataset/mvs_training/dtu";
//path for pair.txt
const std::string pair_file_path = dataset_base_path + "/Cameras/pair.txt";
//path for camera information
//const std::string camera_path = dataset_base_path + "Cameras/train";
//path for depth image
//const std::string depth_image_path = dataset_base_path +


//a struct for a single image.
struct Image_item {
    // best pair number of current pair
    std::vector<int> pairs;
    // information from *_cam.txt
    double extrinsic_mat[4][4];
    double intrinsic_mat[3][3];
    float depth_min;
    float depth_interval;
    // current image data
    std::string image_path;
    // current depth data
    std::string depth_path;
    std::string depth_pfm_path;

};

class Dataloader {
public:
    void setNum_scan(int);
    void setIndex_pose(int);
    void setNum_pose(int);
    int getNum_scan() const;
    int getIndex_pose() const;
    int getNum_pose() const;
    void retrieveItem();
    Image_item getItem() const;

    explicit Dataloader(int num_scan = 1, int num_pose = 1, int index_pose = 0);

private:
    int num_scan;
    int num_pose;
    int index_pose;
    Image_item imageItem;


};


#endif //STEREO_RECONSTRUCTION_DATALOADER_DTU_H
