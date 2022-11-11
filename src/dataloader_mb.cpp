//
// Created by Li Zhou on 22-10-24.
//

#include "dataloader_mb.h"

#include <utility>
#include <fstream>
#include <sstream>

using namespace std;

Dataloader::Dataloader() = default;

Dataloader::Dataloader(string name) {
    this->dataset_name = std::move(name);
}

void Dataloader::setDataset_name(string name) {
    this->dataset_name = std::move(name);
}

string Dataloader::getDataset_name() const {
    return this->dataset_name;
}

void Dataloader::retrievePair() {
    this->imagePair.view_path_0 = dataset_base_path + this->dataset_name + "-perfect/" + "im0.png";
    this->imagePair.view_path_1 = dataset_base_path + this->dataset_name + "-perfect/" + "im1.png";
    this->imagePair.disparity_path_0 = dataset_base_path + this->dataset_name + "-perfect/" + "disp0.pfm";
    this->imagePair.disparity_path_1 = dataset_base_path + this->dataset_name + "-perfect/" + "disp1.pfm";
    // now deal with calib.txt
    string calibfile_path = dataset_base_path + this->dataset_name + "-perfect/" + "calib.txt";
    ifstream calibfile(calibfile_path);
    string line;
    while(getline(calibfile, line)) {
//        cout << line << endl;
        istringstream iss(line);
        string word;
        if(line.rfind("cam0", 0) == 0) {
//            cout << word;
            for(int i = 0; i < 9; i++) {
                iss >> word;
                if(i == 0) {
                    //cout << word.substr(word.find("[") + 1);
                    this->imagePair.intrinsic_mtx0[i / 3][i % 3] = stof(word.substr(word.find("[") + 1));
                } else if(i == 2) {
                    this->imagePair.intrinsic_mtx0[i / 3][i % 3] = stof(word.substr(0, word.length() - 1));
                } else if(i == 5) {
                    this->imagePair.intrinsic_mtx0[i / 3][i % 3] = stof(word.substr(0, word.length() - 1));
                } else if(i == 8) {
                    this->imagePair.intrinsic_mtx0[i / 3][i % 3] = stof(word.substr(0, word.length() - 1));
                } else{
                    this->imagePair.intrinsic_mtx0[i / 3][i % 3] = stof(word);
                }
            }
        } else if(line.rfind("cam1", 0) == 0) { //get the start of each line
            for(int i = 0; i < 9; i++) {
                iss >> word;
                if(i == 0) {
                    this->imagePair.intrinsic_mtx1[i / 3][i % 3] = stof(word.substr(word.find("[") + 1));
                } else if(i == 2) {
                    this->imagePair.intrinsic_mtx1[i / 3][i % 3] = stof(word.substr(0, word.length() - 1));
                } else if(i == 5) {
                    this->imagePair.intrinsic_mtx1[i / 3][i % 3] = stof(word.substr(0, word.length() - 1));
                } else if(i == 8) {
                    this->imagePair.intrinsic_mtx1[i / 3][i % 3] = stof(word.substr(0, word.length() - 1));
                } else{
                    this->imagePair.intrinsic_mtx1[i / 3][i % 3] = stof(word);
                }
            }
        } else if(line.rfind("doffs", 0) == 0) {
            this->imagePair.doffs = stof(line.substr(line.find("=") + 1));
        } else if(line.rfind("baseline", 0) == 0) {
            this->imagePair.baseline = stof(line.substr(line.find("=") + 1));
        } else if(line.rfind("width", 0) == 0) {
            this->imagePair.width = stoi(line.substr(line.find("=") + 1));
        } else if(line.rfind("height", 0) == 0) {
            this->imagePair.height = stoi(line.substr(line.find("=") + 1));
        } else if(line.rfind("vmin", 0) == 0) {
            this->imagePair.vmin = stoi(line.substr(line.find("=") + 1));
        } else if(line.rfind("vmax", 0) == 0) {
            this->imagePair.vmax = stoi(line.substr(line.find("=") + 1));
        }
    }


}

Image_pair Dataloader::getPair() const {
    return this->imagePair;
}
