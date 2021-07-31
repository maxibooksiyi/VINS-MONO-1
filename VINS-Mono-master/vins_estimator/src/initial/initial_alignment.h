#pragma once
#include <eigen3/Eigen/Dense>
#include <iostream>
#include "../factor/imu_factor.h"
#include "../utility/utility.h"
#include <ros/ros.h>
#include <map>
#include "../feature_manager.h"

using namespace Eigen;
using namespace std;


class ImageFrame
{
    public:
        ImageFrame(){};
        ImageFrame(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>& _points, double _t):t{_t},is_key_frame{false}
        {
            points = _points;
        };
        // 图像特征点points,map第一索引是feature_id,内部第二层索引是camera_id(0/1)
        map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>> > > points;
        double t;
        Matrix3d R;  // R^c0_bk  根据sfm得来的
        Vector3d T;  // P^c0_ck
        IntegrationBase *pre_integration;  // 预积分
        bool is_key_frame;
};
// 视觉惯性联合初始化
bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs, Vector3d &g, VectorXd &x);