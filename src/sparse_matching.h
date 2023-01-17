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
    void match(cv::Mat, cv::Mat, std::vector<cv::KeyPoint>, std::vector<cv::KeyPoint>);
    void ransac(std::vector<cv::KeyPoint>, std::vector<cv::KeyPoint>);
    std::vector<cv::DMatch> getGood_matches();
    std::vector<cv::DMatch> get_sorted();
    static bool sort_distance(cv::DMatch, cv::DMatch);
    std::vector<cv::Point2f> getMatched0();
    std::vector<cv::Point2f> getMatched1();

private:
    int mode;
    int norm_type;
    std::vector<cv::DMatch> good_matches;
    std::vector<cv::Point2f> key_points0;
    std::vector<cv::Point2f> key_points1;
    cv::Ptr<cv::DescriptorMatcher> matcher;


};


#endif //STEREO_RECONSTRUCTION_SPARSE_MATCHING_H
