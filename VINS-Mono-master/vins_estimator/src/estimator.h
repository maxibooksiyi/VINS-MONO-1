#pragma once

#include "parameters.h"
#include "feature_manager.h"
#include "utility/utility.h"
#include "utility/tic_toc.h"
#include "initial/solve_5pts.h"
#include "initial/initial_sfm.h"
#include "initial/initial_alignment.h"
#include "initial/initial_ex_rotation.h"
#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>

#include <ceres/ceres.h>
#include "factor/imu_factor.h"
#include "factor/pose_local_parameterization.h"
#include "factor/projection_factor.h"
#include "factor/projection_td_factor.h"
#include "factor/marginalization_factor.h"

#include <unordered_map>
#include <queue>
#include <opencv2/core/eigen.hpp>


class Estimator
{
  public:
    Estimator();

    void setParameter();
    // interface
    void processIMU(double t, const Vector3d &linear_acceleration, const Vector3d &angular_velocity);
    void processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const std_msgs::Header &header);
    void setReloFrame(double _frame_stamp, int _frame_index, vector<Vector3d> &_match_points, Vector3d _relo_t, Matrix3d _relo_r);

    // internal
    void clearState();
    // 视觉初始化入口
    bool initialStructure();
    //
    bool visualInitialAlign();
    // 相对位置姿态
    bool relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l);
    void slideWindow();
    void solveOdometry();
    void slideWindowNew();
    void slideWindowOld();
    void optimization();
    void vector2double();  //
    void double2vector();  //后端优化后的变量更新
    bool failureDetection();

    enum SolverFlag  //
    {
        INITIAL,
        NON_LINEAR
    };

    enum MarginalizationFlag
    {
        MARGIN_OLD = 0,
        MARGIN_SECOND_NEW = 1
    };

    SolverFlag solver_flag; //
    MarginalizationFlag  marginalization_flag;
    Vector3d g;
    MatrixXd Ap[2], backup_A;
    VectorXd bp[2], backup_b;

    Matrix3d ric[NUM_OF_CAM]; //
    Vector3d tic[NUM_OF_CAM]; //
    // 要优化  优化时转成double
    /*
    数据结构:
    1、Rs[frame_count]，Ps[frame_count]，Vs[frame_count]:是从IMU系转到world系的PVQ，数据是由IMU预积分得到的，目前在这里存放的是没有用bias修正过的值。
    2、frame_count:这个值让我很疑惑，它只在processImage()里有过++操作，而且在estimator.hpp声明的时候，没有加上static关键字。它是在h文件中声明，在cpp文件里初始化的，后续需要再关注一下。
    3、dt_buf，linear_acceleration_buf，angular_velocity_buf：帧数和IMU测量值的缓存，而且它们是对齐的。
    3、pre_integrations[frame_count]，它是IntegrationBase的一个实例，在factor/integration_base.h中定义，它保存着frame_count帧中所有跟IMU预积分相关的量，包括F矩阵，Q矩阵，J矩阵等。
    */

    Vector3d Ps[(WINDOW_SIZE + 1)];   //位置  世界坐标系下的
    Vector3d Vs[(WINDOW_SIZE + 1)];   //速度
    Matrix3d Rs[(WINDOW_SIZE + 1)];   //姿态
    Vector3d Bas[(WINDOW_SIZE + 1)];  //角速度计偏差   机体系下的
    Vector3d Bgs[(WINDOW_SIZE + 1)];  //陀螺漂移
    double td; //

    Matrix3d back_R0, last_R, last_R0;
    Vector3d back_P0, last_P, last_P0;
    std_msgs::Header Headers[(WINDOW_SIZE + 1)];  //

    IntegrationBase *pre_integrations[(WINDOW_SIZE + 1)]; //滑窗大小+1
    Vector3d acc_0, gyr_0; //

    vector<double> dt_buf[(WINDOW_SIZE + 1)];  //
    vector<Vector3d> linear_acceleration_buf[(WINDOW_SIZE + 1)]; //
    vector<Vector3d> angular_velocity_buf[(WINDOW_SIZE + 1)];  //

    int frame_count;  //
    int sum_of_outlier, sum_of_back, sum_of_front, sum_of_invalid;

    FeatureManager f_manager;  //
    MotionEstimator m_estimator;   //
    InitialEXRotation initial_ex_rotation;  //

    bool first_imu;  //
    bool is_valid, is_key;
    bool failure_occur;

    vector<Vector3d> point_cloud;
    vector<Vector3d> margin_cloud;
    vector<Vector3d> key_poses;
    double initial_timestamp;


    double para_Pose[WINDOW_SIZE + 1][SIZE_POSE];  // 位置 姿态
    double para_SpeedBias[WINDOW_SIZE + 1][SIZE_SPEEDBIAS];   // 速度 偏差
    double para_Feature[NUM_OF_F][SIZE_FEATURE]; // 逆深度
    double para_Ex_Pose[NUM_OF_CAM][SIZE_POSE];  // 相机到IMU的平移和旋转
    double para_Retrive_Pose[SIZE_POSE];
    double para_Td[1][1]; // IMU 与相机时间差
    double para_Tr[1][1];

    int loop_window_index;

    MarginalizationInfo *last_marginalization_info;
    vector<double *> last_marginalization_parameter_blocks; //

    map<double, ImageFrame> all_image_frame;  //
    IntegrationBase *tmp_pre_integration;  //

    //relocalization variable
    bool relocalization_info;
    double relo_frame_stamp;
    double relo_frame_index;
    int relo_frame_local_index;   // 回环帧在滑窗中的位置
    vector<Vector3d> match_points;
    double relo_Pose[SIZE_POSE];
    Matrix3d drift_correct_r;  // 优化后的两个世界坐标系的位姿
    Vector3d drift_correct_t;  //
    Vector3d prev_relo_t;
    Matrix3d prev_relo_r;
    Vector3d relo_relative_t;
    Quaterniond relo_relative_q;
    double relo_relative_yaw;
};
