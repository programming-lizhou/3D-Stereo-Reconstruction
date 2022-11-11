//
// Created by Li Zhou on 22-10-21.
//

#include "dataloader_dtu.h"
#include <fstream>
#include <sstream>

using namespace std;

Dataloader::Dataloader(int num_scan, int num_pose, int index_pose) {
    this->num_scan = num_scan;
    this->num_pose = num_pose;
    this->index_pose = index_pose;
}

void Dataloader::retrieveItem() {
//first, we deal with pair.txt and get all pairs.
    ifstream pairfile(pair_file_path);
    string str;
    // go to the line which contains the pairs, we get str of the line
    for(int i = 0; i < 2 * getNum_pose() + 1; i++) {
        getline(pairfile, str);
    }
    //then we get the 10 pair numbers
    istringstream iss(str);
    string word;
    int i = 0;
    while(iss >> word) {
        if(i % 2) {
            imageItem.pairs.push_back(stoi(word));
        }
        ++i;
    }
    //then deal with camera data
    //first we get path
    string npose;
    if(getNum_pose() <= 10) {
        npose = "0" + to_string(getNum_pose() - 1);
    } else {
        npose = to_string(getNum_pose() - 1);
    }
    string camera_path = dataset_base_path + "/Cameras/train" + "/000000" + npose + "_cam.txt";
    //then deal with file
    ifstream camerafile(camera_path);
    string line;
    int line_no = 1; // record line number
    while(getline(camerafile, line)) {
        istringstream iss(line);
        string num;
        if(line_no >= 2 && line_no <= 5) {
            for(int j = 0; j < 4; j++) {
                iss >> num;
                imageItem.extrinsic_mat[line_no - 2][j] = stod(num);
            }
        }
        if(line_no >= 8 && line_no <= 10) {
            for(int j = 0; j < 3; j++) {
                iss >> num;
                imageItem.intrinsic_mat[line_no - 8][j] = stod(num);
            }
        }
        if(line_no == 12) {
            iss >> num;
            imageItem.depth_min = stof(num);
            iss >> num;
            imageItem.depth_interval = stof(num);
        }
        ++line_no;
    }
    //finally, deal with the image file and depth file, we simply record the path
    if(getNum_pose() < 10) {
        npose = "0" + to_string(getNum_pose() - 1);
    } else {
        npose = to_string(getNum_pose() - 1);
    }

    imageItem.image_path = dataset_base_path + "/Rectified/scan" + to_string(getNum_scan()) + "_train/" + "rect_0" + npose +
            "_" + to_string(getIndex_pose()) + "_r5000.png";

    if(getNum_pose() <= 10) {
        npose = "0" + to_string(getNum_pose() - 1);
    } else {
        npose = to_string(getNum_pose() - 1);
    }
    imageItem.depth_path = dataset_base_path + "/Depths/scan" + to_string(getNum_scan()) + "_train/" + "depth_visual_00" +
            npose + ".png";
    imageItem.depth_pfm_path = dataset_base_path + "/Depths/scan" + to_string(getNum_scan()) + "_train/" + "depth_map_00" +
                           npose + ".pfm";

}


int Dataloader::getNum_scan() const{
    return num_scan;
};

int Dataloader::getIndex_pose() const{
    return index_pose;
}

int Dataloader::getNum_pose() const{
    return num_pose;
}

Image_item Dataloader::getItem() const{
    return imageItem;
}

void Dataloader::setIndex_pose(int index) {
    this->index_pose = index;
}

void Dataloader::setNum_scan(int num) {
    this->num_scan = num;
}

void Dataloader::setNum_pose(int num) {
    this->num_pose = num;
}