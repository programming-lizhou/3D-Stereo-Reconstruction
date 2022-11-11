//
// Created by Li Zhou on 22-10-24.
//

#include "sparse_matching.h"

using namespace std;
using namespace cv;




SparseMatching::SparseMatching() = default;

SparseMatching::SparseMatching(int md, int norm) {
    this->mode = md;
//    this->num_want = num;
    this->norm_type = norm;
}

void SparseMatching::match(Mat descriptor0, Mat descriptor1, float ratio_thresh) {
    if(mode == 0) {
        matcher = BFMatcher::create(this->norm_type);
        vector<DMatch> all_matches;
        matcher->match(descriptor0, descriptor1, all_matches);

        // compute min and max distance
        double min_dist = 9999;
        double max_dist = 0;
        for(int i = 0; i < all_matches.size(); i++) {
            double dist = all_matches[i].distance;
            min_dist = (dist <= min_dist)?dist:min_dist;
            max_dist = (dist >= max_dist)?dist:max_dist;
        }

        for(int i = 0; i < all_matches.size(); i++) {
            // filter
            if(all_matches[i].distance < 4 * min_dist) {
                this->good_matches.push_back(all_matches[i]);
            }
        }

    } else if(mode == 1) {
        matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
        vector<vector<DMatch> > knn_matches;
        matcher->knnMatch(descriptor0, descriptor1, knn_matches, 2);
        //filter the matches, with lowe's ratio test
//        const float ratio_thresh = 0.4f;
        //with the help of opencv docutment
        for(int i = 0; i < knn_matches.size(); i++) {
            if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
            {
                good_matches.push_back(knn_matches[i][0]);
            }
        }

    }
    /*
    else if(mode == 2) { //thanks to blog https://blog.csdn.net/sinat_41686583/article/details/115186277
        matcher = BFMatcher::create(this->norm_type);
        vector<DMatch> all_matches;
        matcher->match(descriptor0, descriptor1, all_matches);
        vector<Point2f> srcpts(all_matches.size());
        vector<Point2f> dstpts(all_matches.size());
        for(int i = 0; i < all_matches.size(); i++) {
            srcpts.push_back(keypoints0[all_matches[i].queryIdx].pt);
            dstpts.push_back(keypoints1[all_matches[i].trainIdx].pt);
        }
        int reprojectionThreshold = 3;
        vector<unsigned char> mask(srcpts.size());
        Mat mtx = findHomography(srcpts, dstpts, RANSAC, reprojectionThreshold, mask);
        for(int i = 0; i < mask.size(); i++) {
            cout << (int)mask[i];
            if((int)mask[i]) {
                this->good_matches.push_back(all_matches[i]);
            }
        }
        all_matches.swap(this->good_matches);
        cout << all_matches.size();
        cout << good_matches.size();
    } */
}

void SparseMatching::ransac(vector<KeyPoint> keypoints0, vector<KeyPoint> keypoints1) {
    //ransac method, remove outliers.
    //reference: https://blog.csdn.net/Winder_Sky/article/details/79393198?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2~default~CTRLIST~Rate-1-79393198-blog-109079018.pc_relevant_3mothn_strategy_recovery&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2~default~CTRLIST~Rate-1-79393198-blog-109079018.pc_relevant_3mothn_strategy_recovery&utm_relevant_index=2
    //reference: https://blog.csdn.net/sinat_41686583/article/details/115186277
    if(this->good_matches.size() > 10) {
        vector<DMatch> good_matches_origin(this->good_matches);
//        cout << good_matches_origin.size() << endl;
        vector<Point2f> srcpts(good_matches_origin.size());
        vector<Point2f> dstpts(good_matches_origin.size());
        for (int i = 0; i < good_matches_origin.size(); i++) {
            srcpts[i] = keypoints0[good_matches_origin[i].queryIdx].pt;
            dstpts[i] = keypoints1[good_matches_origin[i].trainIdx].pt;
        }
        int reprojectionThreshold = 3;
        vector<unsigned char> mask;
//        cout << srcpts << endl;
        Mat mtx = findHomography(srcpts, dstpts, RANSAC, reprojectionThreshold, mask);
//        cout << mask.size() << endl;
        this->good_matches.clear();
        for (int i = 0; i < mask.size(); i++) {
//            cout << (int) mask[i];
            if ((int) mask[i]) {
                this->good_matches.push_back(good_matches_origin[i]);
            }
        }
//        cout << good_matches_origin.size();
//        cout << good_matches.size();
    }
}

bool SparseMatching::sort_distance(DMatch dMatch1, DMatch dMatch2) {
    return dMatch1.distance < dMatch2.distance;
}


vector<DMatch> SparseMatching::get_sorted() {
    vector<DMatch> matches(this->good_matches);
    sort(matches.begin(), matches.end(), sort_distance);
    return matches;
}




vector<cv::DMatch> SparseMatching::getGood_matches() {
    return this->good_matches;
}

//we need
