//
// Created by lz on 22-10-28.
//

#include "bundle_adjustment.h"

#include <utility>


BA::BA(Image_pair ip, std::vector<cv::Point2f> points1, std::vector<cv::Point2f> points2) {
    this->imagePair = ip;

    this->points1 = points1;
    this->points2 = points2;

    // linear solver
    typedef g2o::BlockSolver< g2o::BlockSolverTraits<6,3> > Block; // dim of pose is 6, dim of landmark is 3
    std::unique_ptr<Block::LinearSolverType> linearSolver (new g2o::LinearSolverCSparse<Block::PoseMatrixType>());
    std::unique_ptr<Block> solver_ptr (new Block ( std::move(linearSolver)));
    // L-M
    auto algorithm = new g2o::OptimizationAlgorithmLevenberg (std::move(solver_ptr));

    this->optimizer.setAlgorithm(algorithm);
    this->optimizer.setVerbose(false);
}

std::pair<cv::Mat, cv::Mat> BA::optimize(std::pair<cv::Mat, cv::Mat>& Transformation, int iteration) {
    // set the first camera vertex
    g2o::VertexSE3Expmap* camera_vertex_1 = new g2o::VertexSE3Expmap();
    camera_vertex_1->setEstimate(g2o::SE3Quat());
    camera_vertex_1->setId(0);
    camera_vertex_1->setFixed(true);
    this->optimizer.addVertex(camera_vertex_1);

    // set the second camera vertex
    g2o::VertexSE3Expmap* camera_vertex_2 = new g2o::VertexSE3Expmap();
    // get initialized Rotation and Translation matrix
    cv::Mat Rotation = Transformation.first;
    cv::Mat Translation = Transformation.second;
    g2o::Matrix3 R_init;
    g2o::Vector3 T_init;
    R_init << Rotation.at<double>(0, 0), Rotation.at<double>(0, 1), Rotation.at<double>(0, 2),
            Rotation.at<double>(1, 0), Rotation.at<double>(1, 1), Rotation.at<double>(1, 2),
            Rotation.at<double>(2, 0), Rotation.at<double>(2, 1), Rotation.at<double>(2, 2);
    T_init << Translation.at<double>( 0,0), Translation.at<double>( 1,0), Translation.at<double>(2,0);
    camera_vertex_2->setEstimate(g2o::SE3Quat(R_init, T_init)); // based on previous results
    camera_vertex_2->setId(1);
    this->optimizer.addVertex(camera_vertex_2);

    // set point vertices based on the first keyframe
    for (size_t i=0;i<this->points1.size();i++){
        g2o::VertexPointXYZ* point_vertex = new g2o::VertexPointXYZ();
        point_vertex->setId(2 + i); // each vertex should its own number
        double x = (this->points1[i].x - this->imagePair.intrinsic_mtx0[0][2]) * 1.0 / this->imagePair.intrinsic_mtx0[0][0];
        double y = (this->points1[i].y - this->imagePair.intrinsic_mtx0[1][2]) * 1.0 / this->imagePair.intrinsic_mtx0[0][0];
        point_vertex->setMarginalized(true);
        point_vertex->setEstimate(Eigen::Vector3d(x, y, 1.0));
        this->optimizer.addVertex(point_vertex);
    }

    // set up two cameras parameters
    g2o::CameraParameters* camera1 = new g2o::CameraParameters(this->imagePair.intrinsic_mtx0[0][0], Eigen::Vector2d(this->imagePair.intrinsic_mtx0[0][2], this->imagePair.intrinsic_mtx0[1][2]), 0);
    camera1->setId(0);
    this->optimizer.addParameter(camera1);

    g2o::CameraParameters* camera2 = new g2o::CameraParameters(this->imagePair.intrinsic_mtx1[0][0], Eigen::Vector2d(this->imagePair.intrinsic_mtx1[0][2], this->imagePair.intrinsic_mtx1[1][2]), 0);
    camera2->setId(1);
    this->optimizer.addParameter(camera2);

    // set up edges
    std::vector<g2o::EdgeProjectXYZ2UV*> edges;
    for (int i = 0; i < this->points1.size(); i++)
    {
        g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
        edge->setVertex(0, dynamic_cast<g2o::VertexPointXYZ*> (this->optimizer.vertex(i+2))); // iterate through all the point vertex
        edge->setVertex(1, dynamic_cast<g2o::VertexSE3Expmap*> (this->optimizer.vertex(0))); // first camera
        edge->setMeasurement(Eigen::Vector2d(this->points1[i].x, this->points1[i].y));
        edge->setInformation(Eigen::Matrix2d::Identity());
        edge->setParameterId(0, 0);
        edge->setRobustKernel(new g2o::RobustKernelHuber());
        this->optimizer.addEdge(edge);
        edges.push_back(edge);
    }

    for (int j = 0; j < this->points2.size(); j++)
    {
        g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
        edge->setVertex(0, dynamic_cast<g2o::VertexPointXYZ*> (this->optimizer.vertex(j+2))); // iterate through all the point vertex
        edge->setVertex(1, dynamic_cast<g2o::VertexSE3Expmap*> (this->optimizer.vertex(1))); // second camera
        edge->setMeasurement(Eigen::Vector2d(this->points2[j].x, this->points2[j].y));
        edge->setInformation(Eigen::Matrix2d::Identity());
        edge->setParameterId(0, 1);
        edge->setRobustKernel(new g2o::RobustKernelHuber());
        this->optimizer.addEdge(edge);
        edges.push_back(edge);
    }

    // start optimizing
    this->optimizer.setVerbose(false);
    this->optimizer.initializeOptimization();
    this->optimizer.optimize(iteration);

    // get transformation matrix
    g2o::VertexSE3Expmap* v = dynamic_cast<g2o::VertexSE3Expmap*>(this->optimizer.vertex(1)); // vertex(1)->transformation matrix
    this->pose = v->estimate();

    // transform pose into cv::Mat
    // get optimized rotation matrix
    cv::Mat iter_rotation = (cv::Mat_<double>(3, 3) << this->pose.rotation()(0, 0), this->pose.rotation()(0, 1), this->pose.rotation()(0, 2),
                                                                 this->pose.rotation()(1, 0), this->pose.rotation()(1, 1), this->pose.rotation()(1, 2),
                                                                 this->pose.rotation()(2, 0), this->pose.rotation()(2, 1), this->pose.rotation()(2, 2));

    // get optimized translation matrix
    Eigen::Vector3d translation = this->pose.translation() / this->pose.translation().norm();
    cv::Mat iter_translation = (cv::Mat_<double>(3, 1) << translation[0], translation[1], translation[2]);

    // build transformation matrix in a pair of (R,T)
    std::pair iter_transformation = std::make_pair(iter_rotation, iter_translation);
    return iter_transformation;
}

