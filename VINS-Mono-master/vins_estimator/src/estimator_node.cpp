#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "estimator.h"
#include "parameters.h"
#include "utility/visualization.h"


Estimator estimator;

std::condition_variable con;
double current_time = -1;
queue<sensor_msgs::ImuConstPtr> imu_buf;
queue<sensor_msgs::PointCloudConstPtr> feature_buf;
queue<sensor_msgs::PointCloudConstPtr> relo_buf;
int sum_of_wait = 0;   // 等待IMU刷新时间

std::mutex m_buf;
std::mutex m_state;
std::mutex i_buf;
std::mutex m_estimator;

double latest_time;
Eigen::Vector3d tmp_P;
Eigen::Quaterniond tmp_Q;
Eigen::Vector3d tmp_V;
Eigen::Vector3d tmp_Ba;
Eigen::Vector3d tmp_Bg;
Eigen::Vector3d acc_0;
Eigen::Vector3d gyr_0;
bool init_feature = 0;
bool init_imu = 1;
double last_imu_t = 0;


/// 离散中值积分：从IMU测量值imu_msg和上一个IMU时刻pvq递推得到下一个tmp_Q，tmp_P，tmp_V
void predict(const sensor_msgs::ImuConstPtr &imu_msg)
{
    // 1.先取当前imu时间戳
    double t = imu_msg->header.stamp.toSec();
    // 如果imu第一个时间戳
    if (init_imu)
    {
        latest_time = t;
        init_imu = 0;
        return;
    }
    double dt = t - latest_time; // 时间间隔
    latest_time = t; //更新上一次imu时间戳
    // 2.线性加速度和角速度
    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    Eigen::Vector3d linear_acceleration{dx, dy, dz};

    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Eigen::Vector3d angular_velocity{rx, ry, rz};

    Eigen::Vector3d un_acc_0 = tmp_Q * (acc_0 - tmp_Ba) - estimator.g;

    Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - tmp_Bg;  /// 中值积分的角速度
    tmp_Q = tmp_Q * Utility::deltaQ(un_gyr * dt);

    Eigen::Vector3d un_acc_1 = tmp_Q * (linear_acceleration - tmp_Ba) - estimator.g;

    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);   /// 中值积分的加速度

    tmp_P = tmp_P + dt * tmp_V + 0.5 * dt * dt * un_acc;
    tmp_V = tmp_V + dt * un_acc;

    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

/*从估计器中得到滑动窗口当前图像帧的imu更新项[P,Q,V,ba,bg,a,g]
因为imu的频率比图像频率要高很多，在getMeasurements(）将图像和imu时间对齐后，imu_buf中还会存在剩余的imu数据
对imu_buf中剩余的所有imu_msg进行PVQ递推
 */
void update()
{
    TicToc t_predict;
    latest_time = current_time;
    tmp_P = estimator.Ps[WINDOW_SIZE];
    tmp_Q = estimator.Rs[WINDOW_SIZE];
    tmp_V = estimator.Vs[WINDOW_SIZE];
    tmp_Ba = estimator.Bas[WINDOW_SIZE];
    tmp_Bg = estimator.Bgs[WINDOW_SIZE];
    acc_0 = estimator.acc_0;
    gyr_0 = estimator.gyr_0;

    queue<sensor_msgs::ImuConstPtr> tmp_imu_buf = imu_buf;
    for (sensor_msgs::ImuConstPtr tmp_imu_msg; !tmp_imu_buf.empty(); tmp_imu_buf.pop())
        predict(tmp_imu_buf.front());
}

/*
对imu和图像数据对齐组合，直到把缓存中图像数据或者IMU数据读取完，才能跳出函数，返回数据。
注意这里只保证了相邻的feature数据之间有完整的imu数据
主要分为以下几种情况
1.IMU或者相机帧的buff为空，measurements返回空值
2.IMU最新的时间戳imu_buf.back() 小于 最旧的图像帧feature_buf.front()+ td，说明IMU数据太少，等待IMU刷新
3.IMU最旧的时间戳imu_buf.front() 大于 图像帧最旧的时间戳feature_buf.front()+ td，最开始只有图像，过了很久来了IMU，但是第一帧图像太旧了，要丢弃第一帧图像
4.理想：IMU最旧的在图像帧最旧的之前，说明IMU数据足够
 */
std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>>
getMeasurements()
{
    std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;

    while (true)
    {
        // 1.IMU或feature是空的直接返回空结果
        if (imu_buf.empty() || feature_buf.empty())
            return measurements;
        // 2.对齐标准：IMU最后一个数据的时间要大于第一个图像特征数据的时间，否则继续等待，返回空值
        // IMU最新的时间戳 小于 最旧的图像帧，说明IMU数据太少，等待IMU刷新
        if (!(imu_buf.back()->header.stamp.toSec() > feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            //ROS_WARN("wait for imu, only should happen at the beginning");
            sum_of_wait++;
            return measurements;
        }
        // 3.对齐标准：不满足 最旧的IMU在最旧的相机帧之前，即相机帧太旧了，要丢弃,保证了图像帧之前一定要有IMU
        // 开头，只有图像，后来突然来了IMU，但是第一帧图像太旧了，就丢地第一帧图像
        if (!(imu_buf.front()->header.stamp.toSec() < feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            ROS_WARN("throw img, only should happen at the beginning");
            feature_buf.pop();
            continue;
        }
        sensor_msgs::PointCloudConstPtr img_msg = feature_buf.front();
        feature_buf.pop();
        std::vector<sensor_msgs::ImuConstPtr> IMUs;
        // 4.图像数据(img_msg)，对应多组在时间戳内的imu数据,然后塞入measurements
        // 理想：IMU最旧的在图像帧最旧的之前，说明IMU数据足够
        while (imu_buf.front()->header.stamp.toSec() < img_msg->header.stamp.toSec() + estimator.td)
        {
            // emplace_back相比push_back能更好地避免内存的拷贝与移动
            IMUs.emplace_back(imu_buf.front());
            imu_buf.pop();
        }
        IMUs.emplace_back(imu_buf.front());   ///
        if (IMUs.empty())
            ROS_WARN("no imu between two image");
        measurements.emplace_back(IMUs, img_msg);
    }
    return measurements;
}
/*
第一件事就是往imu_buf里放IMU数据，缓存起来；
第二件事就是IMU预积分获得当前时刻的PVQ；
第三件事就是如果当前处于非线性优化阶段的话，需要把第二件事计算得到的PVQ发布到rviz里去，见utility/visualization.cpp的pubLatestOdometry()函数
 */
// 队列imu_buf、feature_buf、relo_buf是被多线程共享的，因而在回调函数将相应的msg放入buf或进行pop时，需要设置互斥锁m_buf，在操作前lock()，操作后unlock()

void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    if (imu_msg->header.stamp.toSec() <= last_imu_t)
    {
        ROS_WARN("imu message in disorder!");
        return;
    }
    last_imu_t = imu_msg->header.stamp.toSec();
    /// 对imu_buf的操作是一个生产者-消费者模型，加入和读取的时候不可以中断，必须加锁以保证数据安全
    m_buf.lock();
    imu_buf.push(imu_msg);
    m_buf.unlock();
    /// 唤醒作用于process线程中的获取观测值数据的函数
    con.notify_one();

    last_imu_t = imu_msg->header.stamp.toSec();

    {
        // 构造互斥锁m_state，析构时解锁
        std::lock_guard<std::mutex> lg(m_state);
        predict(imu_msg);   /// 递推得到IMU的PQV
        std_msgs::Header header = imu_msg->header;
        header.frame_id = "world";
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            pubLatestOdometry(tmp_P, tmp_Q, tmp_V, header);  // 发布最新的由IMU直接递推得到的PQV
    }
}

/*
 * 接收的是feature_tracker_node发布的在cur帧的所有特征点的信息
 * 只干了一件事，就是把cur帧的所有特征点放到feature_buf里，同样需要上锁
 * 注意:cur帧的所有特征点都是整合在一个数据里的，也就是sensor_msgs::PointCloudConstPtr &feature_msg
 */
void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    // 1、 判断是否第一帧，光流没有速度
    if (!init_feature)
    {
        //skip the first detected feature, which doesn't contain optical flow speed
        init_feature = 1;
        return;
    }
    // 2.feature_msg加入到feature_buf中，记得加锁解锁
    m_buf.lock();
    feature_buf.push(feature_msg);
    m_buf.unlock();
    // 3、唤醒作用于process线程中的获取观测值数据的函数
    con.notify_one();
}
/*
 * 干了一件事，就是把所有状态量归零，把buf里的数据全部清
 */
void restart_callback(const std_msgs::BoolConstPtr &restart_msg)
{
    if (restart_msg->data == true)
    {
        ROS_WARN("restart the estimator!");
        // 1.清空feature和imu的buf，记得对m_buf加锁解锁
        m_buf.lock();
        while(!feature_buf.empty())
            feature_buf.pop();
        while(!imu_buf.empty())
            imu_buf.pop();
        m_buf.unlock();
        // 2.估计器重置，参数重置，记得对m_eastimator加锁解锁
        m_estimator.lock();
        estimator.clearState();
        estimator.setParameter();
        m_estimator.unlock();
        current_time = -1;
        last_imu_t = 0;
    }
    return;
}

// 重定位回调函数，将points_msg放入relo_buf
void relocalization_callback(const sensor_msgs::PointCloudConstPtr &points_msg)
{
    //printf("relocalization callback! \n");
    m_buf.lock();
    relo_buf.push(points_msg);
    m_buf.unlock();
}

/// thread: visual-inertial odometry
void process()
{
    while (true)
    {
        /*
        数据结构: measurements
        1、首先，measurements他自己就是一个vector；
        2、对于measurements中的每一个measurement，又由2部分组成；
        3、第一部分，由sensor_msgs::ImuConstPtr组成的vector；
        4、第二部分，一个sensor_msgs::PointCloudConstPtr；
        5、这两个sensor_msgs见3.1-6部分介绍。
        6、为什么要这样配对(一个PointCloudConstPtr配上若干个ImuConstPtr)？
        因为IMU的频率比视觉帧的发布频率要高，所以说在这里，需要把一个视觉帧和之前的一串IMU帧的数据配对起来。
        */
        std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;
        std::unique_lock<std::mutex> lk(m_buf);
        /*容器measurements有了，接下来就是配对IMU和视觉帧的数据，放到容器里去。配对过程也需要上锁，而且是一个条件锁。作者在这里用了一个lambda表达式，
         * 也就是说，在return里面的部分是false时，保持上锁状态，继续配对数据；如果return里面时true，说明配对完成，释放锁，measurements完成了，以供后续使用
         * */
        con.wait(lk, [&]
                 {
            return (measurements = getMeasurements()).size() != 0;
                 });
        lk.unlock();
        m_estimator.lock();
        for (auto &measurement : measurements)
        {
            auto img_msg = measurement.second;
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
            for (auto &imu_msg : measurement.first)
            {
                double t = imu_msg->header.stamp.toSec();
                double img_t = img_msg->header.stamp.toSec() + estimator.td;
                if (t <= img_t)
                { 
                    if (current_time < 0)
                        current_time = t;
                    double dt = t - current_time;
                    ROS_ASSERT(dt >= 0);
                    current_time = t;
                    dx = imu_msg->linear_acceleration.x;
                    dy = imu_msg->linear_acceleration.y;
                    dz = imu_msg->linear_acceleration.z;
                    rx = imu_msg->angular_velocity.x;
                    ry = imu_msg->angular_velocity.y;
                    rz = imu_msg->angular_velocity.z;
                    estimator.processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    //printf("imu: dt:%f a: %f %f %f w: %f %f %f\n",dt, dx, dy, dz, rx, ry, rz);

                }
                else
                {
                    double dt_1 = img_t - current_time;
                    double dt_2 = t - img_t;
                    current_time = img_t;
                    ROS_ASSERT(dt_1 >= 0);
                    ROS_ASSERT(dt_2 >= 0);
                    ROS_ASSERT(dt_1 + dt_2 > 0);
                    double w1 = dt_2 / (dt_1 + dt_2);
                    double w2 = dt_1 / (dt_1 + dt_2);
                    dx = w1 * dx + w2 * imu_msg->linear_acceleration.x;
                    dy = w1 * dy + w2 * imu_msg->linear_acceleration.y;
                    dz = w1 * dz + w2 * imu_msg->linear_acceleration.z;
                    rx = w1 * rx + w2 * imu_msg->angular_velocity.x;
                    ry = w1 * ry + w2 * imu_msg->angular_velocity.y;
                    rz = w1 * rz + w2 * imu_msg->angular_velocity.z;
                    estimator.processIMU(dt_1, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    //printf("dimu: dt:%f a: %f %f %f w: %f %f %f\n",dt_1, dx, dy, dz, rx, ry, rz);
                }
            }
            // set relocalization frame
            //在relo_buf中取出最后一个重定位帧，拿出其中的信息并执行setReloFrame()
            sensor_msgs::PointCloudConstPtr relo_msg = NULL;
            while (!relo_buf.empty())
            {
                relo_msg = relo_buf.front();
                relo_buf.pop();
            }
            if (relo_msg != NULL)
            {
                vector<Vector3d> match_points;
                double frame_stamp = relo_msg->header.stamp.toSec();
                for (unsigned int i = 0; i < relo_msg->points.size(); i++)
                {
                    Vector3d u_v_id;
                    u_v_id.x() = relo_msg->points[i].x;
                    u_v_id.y() = relo_msg->points[i].y;
                    u_v_id.z() = relo_msg->points[i].z;
                    match_points.push_back(u_v_id);
                }
                Vector3d relo_t(relo_msg->channels[0].values[0], relo_msg->channels[0].values[1], relo_msg->channels[0].values[2]);
                Quaterniond relo_q(relo_msg->channels[0].values[3], relo_msg->channels[0].values[4], relo_msg->channels[0].values[5], relo_msg->channels[0].values[6]);
                Matrix3d relo_r = relo_q.toRotationMatrix();
                int frame_index;
                frame_index = relo_msg->channels[0].values[7];
                estimator.setReloFrame(frame_stamp, frame_index, match_points, relo_t, relo_r);
            }

            ROS_DEBUG("processing vision data with stamp %f \n", img_msg->header.stamp.toSec());

            TicToc t_s;
            //建立每个特征点的(camera_id,[x,y,z,u,v,vx,vy])s的map，索引为feature_id
            map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> image;  ///
            for (unsigned int i = 0; i < img_msg->points.size(); i++)
            {
                int v = img_msg->channels[0].values[i] + 0.5;
                int feature_id = v / NUM_OF_CAM;
                int camera_id = v % NUM_OF_CAM;
                //获取img_msg中第i个点的x,y,z坐标，这个是归一化后的坐标值
                double x = img_msg->points[i].x;
                double y = img_msg->points[i].y;
                double z = img_msg->points[i].z;
                double p_u = img_msg->channels[1].values[i];
                double p_v = img_msg->channels[2].values[i];
                //获取像素点在x,y方向上的速度
                double velocity_x = img_msg->channels[3].values[i];
                double velocity_y = img_msg->channels[4].values[i];
                ROS_ASSERT(z == 1);
                Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
                xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
                image[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
            }
            estimator.processImage(image, img_msg->header);

            double whole_t = t_s.toc();
            printStatistics(estimator, whole_t);
            std_msgs::Header header = img_msg->header;
            header.frame_id = "world";

            pubOdometry(estimator, header);
            pubKeyPoses(estimator, header);
            pubCameraPose(estimator, header);
            pubPointCloud(estimator, header);
            pubTF(estimator, header);
            pubKeyframe(estimator);
            if (relo_msg != NULL)
                pubRelocalization(estimator);
            //ROS_ERROR("end: %f, at %f", img_msg->header.stamp.toSec(), ros::Time::now().toSec());
        }
        m_estimator.unlock();
        m_buf.lock();
        m_state.lock();
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            update();
        m_state.unlock();
        m_buf.unlock();
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "vins_estimator");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    readParameters(n);
    estimator.setParameter();
#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif
    ROS_WARN("waiting for image and imu...");

    registerPub(n);

    ros::Subscriber sub_imu = n.subscribe(IMU_TOPIC, 2000, imu_callback, ros::TransportHints().tcpNoDelay());
    ros::Subscriber sub_image = n.subscribe("/feature_tracker/feature", 2000, feature_callback);
    ros::Subscriber sub_restart = n.subscribe("/feature_tracker/restart", 2000, restart_callback);
    ros::Subscriber sub_relo_points = n.subscribe("/pose_graph/match_points", 2000, relocalization_callback);

    std::thread measurement_process{process};   /// 创建VIo主线程process
    ros::spin();

    return 0;
}
