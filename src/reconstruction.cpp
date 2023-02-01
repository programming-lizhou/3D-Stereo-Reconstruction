//
// Created by lz on 22-10-28.
//

//Note: the Vertex structure, mesh generation part in this file reference to 3D Scanning and Motion Capture course Exercise1, also the Eigen.h is from the course too.

#include "reconstruction.h"

#include <utility>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <array>
#include <string>
#include "Eigen.h"

struct Vertex
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	// position stored as 4 floats (4th component is supposed to be 1.0)
	Vector4f position;
	// color stored as 4 unsigned char
	Vector4uc color;
};


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
	//remove nah values, very important
	cv::patchNaNs(depth_norm, 0.0);
    depth_norm *= 255.0;
    this->depth_map = depth_norm;
}



//helper function, to check whether a triangle is valid under certain condition,only write triangles with valid vertices and an edge length smaller than edgeThreshold
bool isValid(Vertex* vertices, int a, int b, int c, float eTS) {
	//check whether coordinate is MINF
	if(vertices[a].position.x() != MINF && vertices[b].position.x() != MINF && vertices[c].position.x() != MINF) {
		//check threshold
		if((vertices[a].position - vertices[b].position).norm() < eTS &&
		   (vertices[a].position - vertices[c].position).norm() < eTS &&
		   (vertices[b].position - vertices[c].position).norm() < eTS) {
			   return true;
			   //std::cout << "test" << std::endl;
		   }
	}
	//std::cout << "test" << std::endl;
	return false;

}


bool Reconstruction::generate_mesh(const std::string& filename) {
    // get intrinsic matrix
    Matrix3f depthIntrinsics;
    for(int i = 0; i < 3; ++i) {
        for(int j = 0; j < 3; ++j) {
            depthIntrinsics(i, j) = this->imagePair.intrinsic_mtx0[i][j];
        }
    }
    Matrix3f depthIntrinsicsInv = depthIntrinsics.inverse();
    float fX = depthIntrinsics(0, 0);
	float fY = depthIntrinsics(1, 1);
	float cX = depthIntrinsics(0, 2);
	float cY = depthIntrinsics(1, 2);

    int nrow = this->depth_map.rows;
    int ncol = this->depth_map.cols;
    Vertex* all_vertices = new Vertex[nrow * ncol];
    std::cout << "nrow: " << nrow << ", ncol: " << ncol << std::endl;
	cv::Mat color_map = cv::imread(this->imagePair.view_path_0);

	std::ofstream ost("output.txt");
	ost << this->depth_map << std::endl;

    for(int i = 0; i < nrow; ++i) {
        for(int j = 0; j < ncol; ++j) {
            int idx = i * ncol + j;
            float depth = this->depth_map.ptr<float>(i)[j] / 255.0;
			//transform to camera coordinate
			float camera_X = (j - cX) / fX * depth;
			float camera_Y = (i - cY) / fY * depth;
			Vector4f camera_4f = Vector4f(camera_X, camera_Y, depth, 1.0);
            
            //all_vertices[idx].position = trajectoryInv * depthExtrinsicsInv * camera_4f;
            all_vertices[idx].position = camera_4f;

            cv::Vec3b color = color_map.at<cv::Vec3b>(i, j);
            all_vertices[idx].color = Vector4uc(color[2], color[1], color[0], 255);

            std::cout << "Point " << i << " " << j << " done" << std::endl;
        }
    }

    // so far we have got the point cloud, then we write the mesh

    float edgeThreshold = 0.01f; // 1cm
    unsigned int nVertices = nrow * ncol;

    unsigned int nFaces = 0;
	//store the faces as vector<string>
    unsigned int total_faces = (nrow-1)*(ncol-1)*2; // a square contains two triangles
    std::vector<std::string> allfaces = std::vector<std::string>(total_faces);
//	std::ostringstream oss;
	for(int i = 0; i < nrow - 1; i++) {
		for(int j = 0; j < ncol - 1; j++) {
			/*
			idx1-----idx2
			|       /   | 
			|     /     |
			|   /       |
			idx4-----idx3
			we need to check 2 triangles : 1 4 2 and 4 3 2

			*/

			int idx1 = i * ncol + j;
			int idx2 = idx1 + 1;
			int idx3 = idx2 + ncol;
			int idx4 = idx3 - 1;
			std::string face;
			if(isValid(all_vertices, idx1, idx2, idx4, edgeThreshold)) {

                allfaces[nFaces] = "3 " + std::to_string(idx1) + " " + std::to_string(idx2) + " " + std::to_string(idx4);
                nFaces++;
//				oss << "3 " << idx1 << " " << idx2 << " " << idx4;

//				oss.clear();
//				std::cout << "t1" << std::endl;
			}
			if(isValid(all_vertices, idx2, idx3, idx4, edgeThreshold)) {

                allfaces[nFaces] = "3 " + std::to_string(idx2) + " " + std::to_string(idx3) + " " + std::to_string(idx4);
//				oss << "3 " << idx2 << " " << idx3 << " " << idx4;
                nFaces++;
//				oss.clear();
//				std::cout << "t2" << std::endl;
			}
            std::cout << "Point " << i << " " << j << " done" << std::endl;
		}
	}

	std::ofstream outFile(filename);
	if (!outFile.is_open()) return false;

	// write header
	outFile << "COFF" << std::endl;

	outFile << "# numVertices numFaces numEdges" << std::endl;

	outFile << nVertices << " " << nFaces << " 0" << std::endl;

	// TODO: save vertices
	for(int i = 0; i < nrow; i++) {
		for(int j = 0; j < ncol; j++) {
			int idx = i * ncol + j;
			if(all_vertices[idx].position.x() == MINF) {
				//save as zero ones
				outFile << "0.0 0.0 0.0 0 0 0 0" << std::endl;
				continue;
			}
			//write vertices
			outFile << all_vertices[idx].position[0] << " "
					<< all_vertices[idx].position[1] << " "
					<< all_vertices[idx].position[2] << " "
					<< (int)all_vertices[idx].color[0] << " "
					<< (int)all_vertices[idx].color[1] << " "
					<< (int)all_vertices[idx].color[2] << " "
					<< (int)all_vertices[idx].color[3] << std::endl;

            std::cout << "Point " << i << " " << j << " done" << ' ' << all_vertices[idx].position[0] <<' ' << (int)all_vertices[idx].color[0]<< std::endl;
		}
	}

	// TODO: save valid faces
	std::cout << "# list of faces" << std::endl;
	std::cout << "# nVerticesPerFace idx0 idx1 idx2 ..." << std::endl;
	//write faces
	for(int i = 0; i < nFaces; i++) {
		outFile << allfaces[i] << std::endl;
	}


	// close file
	outFile.close();

    return true;

}






cv::Mat Reconstruction::get_dmap() {
    return this->depth_map;
}

