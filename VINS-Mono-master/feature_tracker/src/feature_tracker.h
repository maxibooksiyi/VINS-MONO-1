#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

#include "parameters.h"
#include "tic_toc.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;

bool inBorder(const cv::Point2f &pt);

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
void reduceVector(vector<int> &v, vector<uchar> status);

class FeatureTracker
{
  public:
    FeatureTracker();

    void readImage(const cv::Mat &_img,double _cur_time);

    void setMask();
// 添加新检测到的特征点
    void addPoints();

    bool updateID(unsigned int i);
// 读取相机内参数
    void readIntrinsicParameter(const string &calib_file);

    void showUndistortion(const string &name);

    void rejectWithF();
// 将所有特征点转换到相机归一化平面并进行去畸变
    void undistortedPoints();

    cv::Mat mask;  //图像掩码
    cv::Mat fisheye_mask;  //鱼眼相机mask，用来去除边缘噪点
    cv::Mat prev_img, cur_img, forw_img;
    // prev_img是上一次发布的帧的图像数据
    // cur_img是光流跟踪的前一帧的图像数据
    // forw_img是光流跟踪的后一帧的图像数据，也就是当前
    vector<cv::Point2f> n_pts;  //每一帧中新提取的特征点
    vector<cv::Point2f> prev_pts, cur_pts, forw_pts; //对应图像的特征点
    vector<cv::Point2f> prev_un_pts, cur_un_pts;  //归一化相机坐标系下的坐标
    vector<cv::Point2f> pts_velocity;  //前一帧的特征点相对于上一次发布的帧的特征点的沿x,y方向的像素移动速度
    vector<int> ids; //当前帧能够被跟踪到的特征点的id
    vector<int> track_cnt;
    // map<pair<int,cv::Point2f>> cur_un_pts_map ?
    map<int, cv::Point2f> cur_un_pts_map;  //自动建立key-value的对应,key和value可以是任意你需要的类型,key不重复
    map<int, cv::Point2f> prev_un_pts_map;
    camodocal::CameraPtr m_camera;   //相机模型

    double cur_time;
    double prev_time;

    static int n_id;
    //用来作为特征点id，每检测到一个新的特征点，就将n_id作为该特征点的id，然后n_id加1
};
