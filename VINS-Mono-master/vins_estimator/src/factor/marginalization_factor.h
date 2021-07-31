#pragma once

#include <ros/ros.h>
#include <ros/console.h>
#include <cstdlib>
#include <pthread.h>
#include <ceres/ceres.h>
#include <unordered_map>

#include "../utility/utility.h"
#include "../utility/tic_toc.h"

const int NUM_THREADS = 4;

struct ResidualBlockInfo
{
    ResidualBlockInfo(ceres::CostFunction *_cost_function, ceres::LossFunction *_loss_function, std::vector<double *> _parameter_blocks, std::vector<int> _drop_set)
        : cost_function(_cost_function), loss_function(_loss_function), parameter_blocks(_parameter_blocks), drop_set(_drop_set) {}

    void Evaluate();

    ceres::CostFunction *cost_function;
    ceres::LossFunction *loss_function;
    std::vector<double *> parameter_blocks;  //优化变量数据
    std::vector<int> drop_set;  //待边缘化的优化变量id

    double **raw_jacobians;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;
    Eigen::VectorXd residuals; //残差,IMU:15×1,视觉:2×1

    int localSize(int size)
    {
        return size == 7 ? 6 : size;
    }
};

struct ThreadsStruct
{
    std::vector<ResidualBlockInfo *> sub_factors;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    std::unordered_map<long, int> parameter_block_size; //global size
    std::unordered_map<long, int> parameter_block_idx; //local size
};

class MarginalizationInfo  //所有类型残差都添加到此中
{
  public:
    ~MarginalizationInfo();
    int localSize(int size) const;
    int globalSize(int size) const;

    // 添加残差块相关信息(优化变量,待边缘化变量)
    void addResidualBlockInfo(ResidualBlockInfo *residual_block_info);
    //计算每个残差对应的Jacobian,并更新parameter_block_data
    void preMarginalize();
    //pos为所有变量维度,
    // 多线程构造先验项舒尔补AX=b的结构，计算Jacobian和残差
    void marginalize();

    std::vector<double *> getParameterBlocks(std::unordered_map<long, double *> &addr_shift);

    std::vector<ResidualBlockInfo *> factors;  //所有的残差项

    // m为要marg掉的变量个数,也就是parameter_block_idx的总localSize,以double为单位,VBias为9,PQ为6,
    // n为要保留下的优化变量的变量个数,n=localSize(parameter_block_size) – m
    int m, n;  //m为需要marg掉的变量个数,n为需要保留的变量个数
    //<优化变量内存地址,local size>
    std::unordered_map<long, int> parameter_block_size; // local size
    int sum_block_size;
    //<待边缘化的优化变量内存地址,在parameter_block_size中的id,以double为单位>
    std::unordered_map<long, int> parameter_block_idx; //
    //<优化变量内存地址,数据>
    std::unordered_map<long, double *> parameter_block_data; //

    std::vector<int> keep_block_size; //边缘化后保留的各优化变量的大小
    std::vector<int> keep_block_idx;  //
    std::vector<double *> keep_block_data; //

    Eigen::MatrixXd linearized_jacobians;  // 边缘化之后从信息矩阵恢复出来雅克比矩阵
    Eigen::VectorXd linearized_residuals;  // 边缘化之后从信息矩阵恢复出来残差向量
    const double eps = 1e-8;

};

class MarginalizationFactor : public ceres::CostFunction
{
  public:
    MarginalizationFactor(MarginalizationInfo* _marginalization_info);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    MarginalizationInfo* marginalization_info;
};
