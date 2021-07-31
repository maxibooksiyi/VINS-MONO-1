#pragma once

#include <vector>
using namespace std;

#include <opencv2/opencv.hpp>
//#include <opencv2/core/eigen.hpp>
#include <eigen3/Eigen/Dense>
using namespace Eigen;

#include <ros/console.h>

class MotionEstimator
{
  public:
    // 计算匹配的特征点的两帧之间的相对位姿
    // solveRelativeRT()函数 利用五点法求解相机初始位姿
    // estimator.cpp中：计算窗口里最后一帧相对于之前帧的位姿, 这里的R T是在最后一帧(窗口里最后一帧)相对于l帧坐标系（找到的参考帧）的位姿估计
    bool solveRelativeRT(const vector<pair<Vector3d, Vector3d>> &corres, Matrix3d &R, Vector3d &T);

  private:
    double testTriangulation(const vector<cv::Point2f> &l,
                             const vector<cv::Point2f> &r,
                             cv::Mat_<double> R, cv::Mat_<double> t);
    void decomposeE(cv::Mat E,
                    cv::Mat_<double> &R1, cv::Mat_<double> &R2,
                    cv::Mat_<double> &t1, cv::Mat_<double> &t2);
};


