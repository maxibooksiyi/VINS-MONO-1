#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include <list>
#include <algorithm>
#include <vector>
#include <numeric>
using namespace std;

#include <eigen3/Eigen/Dense>
using namespace Eigen;

#include <ros/console.h>
#include <ros/assert.h>

#include "parameters.h"

class FeaturePerFrame  //每个特征点在一张图像中的信息
{
  public:
    FeaturePerFrame(const Eigen::Matrix<double, 7, 1> &_point, double td) //td:IMU与cam同步时间差
    {
        point.x() = _point(0);
        point.y() = _point(1);
        point.z() = _point(2);
        uv.x() = _point(3);
        uv.y() = _point(4);
        velocity.x() = _point(5); 
        velocity.y() = _point(6); 
        cur_td = td;
    }
    double cur_td;  //时间间隔
    Vector3d point;
    Vector2d uv;
    Vector2d velocity;  //修正时像素移动速度
    double z;    //特征点的深度
    bool is_used;  //是否被用了
    double parallax;   //视差
    MatrixXd A;   //变换矩阵
    VectorXd b;
    double dep_gradient;
};

class FeaturePerId
{
  public:
    const int feature_id;  //特征点ID索引
    int start_frame;  //第一次观测到该特征点的图像的id
    vector<FeaturePerFrame> feature_per_frame;   //每个路标点由多个连续的图像观测到,能够观测到某个特征点的所有相关帧

    int used_num;  //该特征出现的次数/作为内点的次数？

    bool is_outlier;  //是否外点
    bool is_margin;   //是否边缘化
    double estimated_depth;  //估计的逆深度
    int solve_flag; // 求解器 0 haven't solve yet; 1 solve succ; 2 solve fail;

    Vector3d gt_p;

    FeaturePerId(int _feature_id, int _start_frame)
        : feature_id(_feature_id), start_frame(_start_frame),
          used_num(0), estimated_depth(-1.0), solve_flag(0)
    {
    }
    // 返回最后一个看到这个特征点的图像帧的id
    int endFrame();
};

class FeatureManager
{
  public:
    FeatureManager(Matrix3d _Rs[]);

    void setRic(Matrix3d _ric[]);

    void clearState();
    // 窗口中被跟踪的特征点数量
    int getFeatureCount();
    // 特征点进入时检查视差,是否为关键帧
    // 图像image,map第一索引是feaure_id,内部第二层索引是camera_id
    //// map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image 参数格式：特征点id 相机id(0/1) 7参数
    //// vector存储左右相机的两张图的特征点的参数
    bool addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td);
    void debugShow();
    // 前后两帧之间匹配特征点的3D坐标
    vector<pair<Vector3d, Vector3d>> getCorresponding(int frame_count_l, int frame_count_r);
    //void updateDepth(const VectorXd &x);
    // 设置特征点的逆深度
    void setDepth(const VectorXd &x);
    void removeFailures();
    void clearDepth(const VectorXd &x);
    VectorXd getDepthVector();
    // 特征点三角化求深度(SVD分解)
    void triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[]);
    //
    void removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P);
    // 边缘化最老帧,直接将特征点保存的帧号前移
    void removeBack();
    // 边缘化次新帧,对特征点在次新帧的信息移除
    void removeFront(int frame_count);
    // 移除外点
    void removeOutlier();
    list<FeaturePerId> feature; //滑窗内所有的特征点(路标点),FeaturePerId:特征点的信息
    int last_track_num; //所有特征点被跟踪到的数目

  private:
    // 计算视差
    double compensatedParallax2(const FeaturePerId &it_per_id, int frame_count);
    const Matrix3d *Rs;
    Matrix3d ric[NUM_OF_CAM];
};

#endif
