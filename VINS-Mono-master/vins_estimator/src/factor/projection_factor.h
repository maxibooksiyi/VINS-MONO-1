#pragma once

#include <ros/assert.h>
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include "../utility/utility.h"
#include "../utility/tic_toc.h"
#include "../parameters.h"

class ProjectionFactor : public ceres::SizedCostFunction<2, 7, 7, 7, 1>  // 因变量2维度 待优化变量维度7 7 7 1
{
  public:
    // //构造函数及切平面空间的基
    ProjectionFactor(const Eigen::Vector3d &_pts_i, const Eigen::Vector3d &_pts_j);
    //视觉测量残差 即 特征点的重投影误差，视觉残差和雅克比矩阵计算的对应代码在 ProjectionFactor::Evaluate 函数中
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;
    void check(double **parameters);

    Eigen::Vector3d pts_i, pts_j;
    Eigen::Matrix<double, 2, 3> tangent_base;
    static Eigen::Matrix2d sqrt_info;
    static double sum_t;
};
