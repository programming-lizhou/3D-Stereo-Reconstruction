//
// Created by Li Zhou on 22-10-24.
//

#ifndef STEREO_RECONSTRUCTION_SPARSE_MATCHING_H
#define STEREO_RECONSTRUCTION_SPARSE_MATCHING_H

#include <vector>
#include <opencv2/opencv.hpp>
#include "opencv2/features2d.hpp"


class SparseMatching {
public:
    SparseMatching();
    SparseMatching(int, int);
    void match(cv::Mat, cv::Mat, float);
    void ransac(std::vector<cv::KeyPoint>, std::vector<cv::KeyPoint>);
    std::vector<cv::DMatch> getGood_matches();
    std::vector<cv::DMatch> get_sorted();
    static bool sort_distance(cv::DMatch, cv::DMatch);
private:
//    int num_want;
    int mode; //0 for brute force + ratio, 1 for flann based + knn lowe, 2 for ransac
    int norm_type;
//    float ratio_thresh;
    std::vector<cv::DMatch> good_matches;
//    std::vector<cv::DMatch> selected_matches;
//    std::vector<cv::DMatch> good_matches_sorted; //sorted by distance of the matched points
    cv::Ptr<cv::DescriptorMatcher> matcher;


};


#endif //STEREO_RECONSTRUCTION_SPARSE_MATCHING_H
