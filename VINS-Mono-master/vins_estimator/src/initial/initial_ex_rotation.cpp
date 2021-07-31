#include "initial_ex_rotation.h"

InitialEXRotation::InitialEXRotation(){
    frame_count = 0;
    Rc.push_back(Matrix3d::Identity());
    Rc_g.push_back(Matrix3d::Identity());
    Rimu.push_back(Matrix3d::Identity());
    ric = Matrix3d::Identity();
}

// corres:匹配的特征点    delta_q_imu:IMU预积分得的旋转矩阵
bool InitialEXRotation::CalibrationExRotation(vector<pair<Vector3d, Vector3d>> corres, Quaterniond delta_q_imu, Matrix3d &calib_ric_result)
{
    frame_count++; //
    Rc.push_back(solveRelativeR(corres));  //计算前后两帧的旋转矩阵,加到Rc向量内 R_bk+1^bk
    Rimu.push_back(delta_q_imu.toRotationMatrix());  // R(bk+1--bk)
    Rc_g.push_back(ric.inverse() * delta_q_imu * ric); //每次迭代前先用前一次估计的ric将R_bk+1^bk变换成R_ck+1^ck
    // SVD分解，Ax=0,对A填充
    Eigen::MatrixXd A(frame_count * 4, 4); // 多帧组成A,一对点组成的A是4*4的
    A.setZero();
    int sum_ok = 0;
    for (int i = 1; i <= frame_count; i++)  //
    {
        // 求解核函数
        Quaterniond r1(Rc[i]);
        Quaterniond r2(Rc_g[i]);
        /* angularDistance就是计算两个坐标系之间相对旋转矩阵在做轴角变换后(u * theta)
        的角度theta, theta越小说明两个坐标系的姿态越接近，这个角度距离用于后面计算权重，
        这里计算权重就是为了降低外点的干扰，意思就是为了防止出现误差非常大的R_bk+1^bk和
        R_ck+1^ck
        这里权重用来抑制过大残差值的影响
        残差是两条路径构建的误差*/
        double angular_distance = 180 / M_PI * r1.angularDistance(r2);
        ROS_DEBUG("%d %f", i, angular_distance);

        double huber = angular_distance > 5.0 ? 5.0 / angular_distance : 1.0;
        ++sum_ok;
        //R_bk+1^bk * R_c^b = R_c^b * R_ck+1^ck
        //[Q(q_bk+1^bk) - Q(q_ck+1^ck)] * q_c^b = 0
        Matrix4d L, R;     ////R_b^c * R_bk+1^bk = R_ck+1^ck *  R_b^c   R为IMU旋转四元数的右乘矩阵  L为相机旋转四元数的左乘矩阵

        double w = Quaterniond(Rc[i]).w();
        Vector3d q = Quaterniond(Rc[i]).vec();
        L.block<3, 3>(0, 0) = w * Matrix3d::Identity() + Utility::skewSymmetric(q);
        L.block<3, 1>(0, 3) = q;
        L.block<1, 3>(3, 0) = -q.transpose();
        L(3, 3) = w;

        Quaterniond R_ij(Rimu[i]);
        w = R_ij.w();
        q = R_ij.vec();
        R.block<3, 3>(0, 0) = w * Matrix3d::Identity() - Utility::skewSymmetric(q);
        R.block<3, 1>(0, 3) = q;
        R.block<1, 3>(3, 0) = -q.transpose();
        R(3, 3) = w;
        // [Q(q_ck+1^ck)-Q(q_bk+1^bk)] * q_b^c  = 0
        A.block<4, 4>((i - 1) * 4, 0) = huber * (L - R);//加入权重huber
    }
    // svd分解中最小奇异值对应的右奇异向量作为旋转四元数
    JacobiSVD<MatrixXd> svd(A, ComputeFullU | ComputeFullV);
    // 这里的四元数存储的顺序是[x,y,z,w]',即[qv qw]'
    Matrix<double, 4, 1> x = svd.matrixV().col(3);
    Quaterniond estimated_R(x);
    ric = estimated_R.toRotationMatrix().inverse();  //求逆得 R_c^b
    //cout << svd.singularValues().transpose() << endl;
    //cout << ric << endl;
    Vector3d ric_cov;
    ric_cov = svd.singularValues().tail<3>();  // svd.singularValues()为奇异值分解对角线元素（从大到小排列）
    // 至少会迭代WINDOW_SIZE次，并且第二小的奇异值大于0.25才认为标定成功
    // 最小的奇异值要足够接近于0，和第二小之间要有足够差距才行
    if (frame_count >= WINDOW_SIZE && ric_cov(1) > 0.25)
    {
        calib_ric_result = ric;
        return true;
    }
    else
        return false;
}

// 计算前后两帧的相机旋转矩阵
Matrix3d InitialEXRotation::solveRelativeR(const vector<pair<Vector3d, Vector3d>> &corres)
{
    if (corres.size() >= 9)  // 需要特征点大于9对，否则返回单位矩阵
    {
        // 归一化相机系下前二维坐标SVD分解
        vector<cv::Point2f> ll, rr;
        for (int i = 0; i < int(corres.size()); i++)
        {
            ll.push_back(cv::Point2f(corres[i].first(0), corres[i].first(1)));
            rr.push_back(cv::Point2f(corres[i].second(0), corres[i].second(1)));
        }
        cv::Mat E = cv::findFundamentalMat(ll, rr);   //求解本质矩阵
        cv::Mat_<double> R1, R2, t1, t2;
        decomposeE(E, R1, R2, t1, t2);  // 本质矩阵E求R t   // 本质矩阵svd分解得到4组Rt解

        if (determinant(R1) + 1.0 < 1e-09)  //determinant:行列式   // 如果行列式为负，SVD分解-E
        {
            E = -E;
            decomposeE(E, R1, R2, t1, t2);
        }
        // 通过三角化得到的正深度选择R t解
        double ratio1 = max(testTriangulation(ll, rr, R1, t1), testTriangulation(ll, rr, R1, t2));
        double ratio2 = max(testTriangulation(ll, rr, R2, t1), testTriangulation(ll, rr, R2, t2));
        cv::Mat_<double> ans_R_cv = ratio1 > ratio2 ? R1 : R2;
        // 对R求转置
        Matrix3d ans_R_eigen;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                ans_R_eigen(j, i) = ans_R_cv(i, j);
        return ans_R_eigen;  // 前一帧相对于后一帧的旋转矩阵   R(ck+1--ck)
    }
    return Matrix3d::Identity();
}

// 三角化测试R t
double InitialEXRotation::testTriangulation(const vector<cv::Point2f> &l,
                                          const vector<cv::Point2f> &r,
                                          cv::Mat_<double> R, cv::Mat_<double> t)
{
    cv::Mat pointcloud;
    // 相机内参
    cv::Matx34f P = cv::Matx34f(1, 0, 0, 0,
                                0, 1, 0, 0,
                                0, 0, 1, 0);
    // 外餐
    cv::Matx34f P1 = cv::Matx34f(R(0, 0), R(0, 1), R(0, 2), t(0),
                                 R(1, 0), R(1, 1), R(1, 2), t(1),
                                 R(2, 0), R(2, 1), R(2, 2), t(2));
    // 但需要注意的是，输出的3D坐标是齐次坐标，共四个维度，因此需要将前三个维度除以第四个维度以得到非齐次坐标xyz
    // 这个坐标是在世界坐标系下的坐标，以输入的两个相机位姿所在的坐标系为准
    cv::triangulatePoints(P, P1, l, r, pointcloud);  // 三角化得到路标3D世界坐标系下的坐标  范湖4xN
    int front_count = 0;
    for (int i = 0; i < pointcloud.cols; i++)
    {
        double normal_factor = pointcloud.col(i).at<float>(3);  //第四维

        cv::Mat_<double> p_3d_l = cv::Mat(P) * (pointcloud.col(i) / normal_factor);  //相机坐标系下的3D坐标
        cv::Mat_<double> p_3d_r = cv::Mat(P1) * (pointcloud.col(i) / normal_factor);
        if (p_3d_l(2) > 0 && p_3d_r(2) > 0)
            front_count++;  //正深度对点数
    }
    ROS_DEBUG("MotionEstimator: %f", 1.0 * front_count / pointcloud.cols);
    return 1.0 * front_count / pointcloud.cols;
}

// 本质矩阵E求R t
void InitialEXRotation::decomposeE(cv::Mat E,
                                 cv::Mat_<double> &R1, cv::Mat_<double> &R2,
                                 cv::Mat_<double> &t1, cv::Mat_<double> &t2)
{
    cv::SVD svd(E, cv::SVD::MODIFY_A); //cv::SVD::MODIFY_A:允许改变矩阵A
    // w:沿着Z轴顺时针旋转90°的旋转矩阵
    cv::Matx33d W(0, -1, 0,
                  1, 0, 0,
                  0, 0, 1);
    // wt:沿着Z轴逆时针旋转90°的旋转矩阵
    cv::Matx33d Wt(0, 1, 0,
                   -1, 0, 0,
                   0, 0, 1);
    R1 = svd.u * cv::Mat(W) * svd.vt;
    R2 = svd.u * cv::Mat(Wt) * svd.vt;
    t1 = svd.u.col(2);
    t2 = -svd.u.col(2);
}
