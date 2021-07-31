#include <vector>
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <visualization_msgs/Marker.h>
#include <std_msgs/Bool.h>
#include <cv_bridge/cv_bridge.h>
#include <iostream>
#include <ros/package.h>
#include <mutex>
#include <queue>
#include <thread>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "keyframe.h"
#include "utility/tic_toc.h"
#include "pose_graph.h"
#include "utility/CameraPoseVisualization.h"
#include "parameters.h"
#define SKIP_FIRST_CNT 10
using namespace std;

queue<sensor_msgs::ImageConstPtr> image_buf;
queue<sensor_msgs::PointCloudConstPtr> point_buf;
queue<nav_msgs::Odometry::ConstPtr> pose_buf;
queue<Eigen::Vector3d> odometry_buf;
std::mutex m_buf;
std::mutex m_process;
int frame_index  = 0;
int sequence = 1;
PoseGraph posegraph;
int skip_first_cnt = 0;
int SKIP_CNT;
int skip_cnt = 0;
bool load_flag = 0;
bool start_flag = 0;
double SKIP_DIS = 0;

int VISUALIZATION_SHIFT_X;
int VISUALIZATION_SHIFT_Y;
int ROW;
int COL;
int DEBUG_IMAGE;
int VISUALIZE_IMU_FORWARD;
int LOOP_CLOSURE;
int FAST_RELOCALIZATION;

camodocal::CameraPtr m_camera;
Eigen::Vector3d tic;
Eigen::Matrix3d qic;
ros::Publisher pub_match_img;
ros::Publisher pub_match_points;
ros::Publisher pub_camera_pose_visual;
ros::Publisher pub_key_odometrys;
ros::Publisher pub_vio_path;
nav_msgs::Path no_loop_path;

std::string BRIEF_PATTERN_FILE;
std::string POSE_GRAPH_SAVE_PATH;
std::string VINS_RESULT_PATH;

CameraPoseVisualization cameraposevisual(1, 0, 0, 1);
Eigen::Vector3d last_t(-100, -100, -100);
double last_image_time = -1;

// 开始一个新的图像序列（地图合并功能)
void new_sequence()
{
    printf("new sequence\n");
    sequence++;
    printf("sequence cnt %d \n", sequence);
    if (sequence > 5)
    {
        ROS_WARN("only support 5 sequences since it's boring to copy code for more sequences.");
        ROS_BREAK();
    }
    posegraph.posegraph_visualization->reset();
    posegraph.publish();
    m_buf.lock();
    while(!image_buf.empty())
        image_buf.pop();
    while(!point_buf.empty())
        point_buf.pop();
    while(!pose_buf.empty())
        pose_buf.pop();
    while(!odometry_buf.empty())
        odometry_buf.pop();
    m_buf.unlock();
}

//图像数据回调函数，将image_msg放入image_buf，同时根据时间戳检测是否是新的图像序列
void image_callback(const sensor_msgs::ImageConstPtr &image_msg)
{
    //ROS_INFO("image_callback!");
    if(!LOOP_CLOSURE)
        return;
    m_buf.lock();
    image_buf.push(image_msg);
    m_buf.unlock();
    //printf(" image time %f \n", image_msg->header.stamp.toSec());

    // detect unstable camera stream
    // 检测不稳定的相机流
    if (last_image_time == -1)
        last_image_time = image_msg->header.stamp.toSec();
    else if (image_msg->header.stamp.toSec() - last_image_time > 1.0 || image_msg->header.stamp.toSec() < last_image_time)
    {
        ROS_WARN("image discontinue! detect a new sequence!");
        new_sequence();
    }
    last_image_time = image_msg->header.stamp.toSec();
}

//地图点云回调函数，把point_msg放入point_buf
void point_callback(const sensor_msgs::PointCloudConstPtr &point_msg)
{
    //ROS_INFO("point_callback!");
    if(!LOOP_CLOSURE)
        return;
    m_buf.lock();
    point_buf.push(point_msg);
    m_buf.unlock();
    /*
    for (unsigned int i = 0; i < point_msg->points.size(); i++)
    {
        printf("%d, 3D point: %f, %f, %f 2D point %f, %f \n",i , point_msg->points[i].x, 
                                                     point_msg->points[i].y,
                                                     point_msg->points[i].z,
                                                     point_msg->channels[i].values[0],
                                                     point_msg->channels[i].values[1]);
    }
    */
}

//图像帧位姿回调函数，把pose_msg放入pose_buf
void pose_callback(const nav_msgs::Odometry::ConstPtr &pose_msg)
{
    //ROS_INFO("pose_callback!");
    if(!LOOP_CLOSURE)
        return;
    m_buf.lock();
    pose_buf.push(pose_msg);
    m_buf.unlock();
    /*
    printf("pose t: %f, %f, %f   q: %f, %f, %f %f \n", pose_msg->pose.pose.position.x,
                                                       pose_msg->pose.pose.position.y,
                                                       pose_msg->pose.pose.position.z,
                                                       pose_msg->pose.pose.orientation.w,
                                                       pose_msg->pose.pose.orientation.x,
                                                       pose_msg->pose.pose.orientation.y,
                                                       pose_msg->pose.pose.orientation.z);
    */
}

// mu前向递推的回调函数，从IMU预积分的位姿得到IMU位姿和cam位姿，得到低延迟和高频率结果
void imu_forward_callback(const nav_msgs::Odometry::ConstPtr &forward_msg)
{
    if (VISUALIZE_IMU_FORWARD)
    {
        Vector3d vio_t(forward_msg->pose.pose.position.x, forward_msg->pose.pose.position.y, forward_msg->pose.pose.position.z);
        Quaterniond vio_q;
        vio_q.w() = forward_msg->pose.pose.orientation.w;
        vio_q.x() = forward_msg->pose.pose.orientation.x;
        vio_q.y() = forward_msg->pose.pose.orientation.y;
        vio_q.z() = forward_msg->pose.pose.orientation.z;

        vio_t = posegraph.w_r_vio * vio_t + posegraph.w_t_vio;
        vio_q = posegraph.w_r_vio *  vio_q;

        vio_t = posegraph.r_drift * vio_t + posegraph.t_drift;
        vio_q = posegraph.r_drift * vio_q;

        Vector3d vio_t_cam;
        Quaterniond vio_q_cam;
        vio_t_cam = vio_t + vio_q * tic;
        vio_q_cam = vio_q * qic;        

        cameraposevisual.reset();
        cameraposevisual.add_pose(vio_t_cam, vio_q_cam);
        cameraposevisual.publish_by(pub_camera_pose_visual, forward_msg->header);
    }
}

// 重定位回调函数，将重定位帧的相对位姿放入loop_info，updateKeyFrameLoop()进行回环更新
void relo_relative_pose_callback(const nav_msgs::Odometry::ConstPtr &pose_msg)
{
    Vector3d relative_t = Vector3d(pose_msg->pose.pose.position.x,
                                   pose_msg->pose.pose.position.y,
                                   pose_msg->pose.pose.position.z);
    Quaterniond relative_q;
    relative_q.w() = pose_msg->pose.pose.orientation.w;
    relative_q.x() = pose_msg->pose.pose.orientation.x;
    relative_q.y() = pose_msg->pose.pose.orientation.y;
    relative_q.z() = pose_msg->pose.pose.orientation.z;
    double relative_yaw = pose_msg->twist.twist.linear.x;
    int index = pose_msg->twist.twist.linear.y;
    //printf("receive index %d \n", index );
    Eigen::Matrix<double, 8, 1 > loop_info;
    loop_info << relative_t.x(), relative_t.y(), relative_t.z(),
                 relative_q.w(), relative_q.x(), relative_q.y(), relative_q.z(),
                 relative_yaw;
    //更新index关键帧的回环信息loop_info
    posegraph.updateKeyFrameLoop(index, loop_info);

}

// VIO回调函数，根据pose_msg中的位姿得到IMU位姿和cam位姿
void vio_callback(const nav_msgs::Odometry::ConstPtr &pose_msg)
{
    //ROS_INFO("vio_callback!");
    Vector3d vio_t(pose_msg->pose.pose.position.x, pose_msg->pose.pose.position.y, pose_msg->pose.pose.position.z);
    Quaterniond vio_q;
    vio_q.w() = pose_msg->pose.pose.orientation.w;
    vio_q.x() = pose_msg->pose.pose.orientation.x;
    vio_q.y() = pose_msg->pose.pose.orientation.y;
    vio_q.z() = pose_msg->pose.pose.orientation.z;
    //w_r_vio默认单位阵、w_t_vio默认零向量
    vio_t = posegraph.w_r_vio * vio_t + posegraph.w_t_vio;
    vio_q = posegraph.w_r_vio *  vio_q;
    //r_drift默认单位阵、t_drift默认零向量，回环检测后值会变化
    vio_t = posegraph.r_drift * vio_t + posegraph.t_drift;
    vio_q = posegraph.r_drift * vio_q; // vio_q vio_t:IMU相对于世界坐标系的位姿

    Vector3d vio_t_cam;
    Quaterniond vio_q_cam;
    //相机在世界坐标系的位姿
    vio_t_cam = vio_t + vio_q * tic;
    vio_q_cam = vio_q * qic;        

    if (!VISUALIZE_IMU_FORWARD)
    {
        //相机位姿发布出去,待订阅后在RVIZ显示，即系统运行时的相框
        cameraposevisual.reset();
        cameraposevisual.add_pose(vio_t_cam, vio_q_cam);
        cameraposevisual.publish_by(pub_camera_pose_visual, pose_msg->header);
    }

    odometry_buf.push(vio_t_cam);
    if (odometry_buf.size() > 10)
    {
        odometry_buf.pop();
    }
    //key_odometrys的一些参数配置工作
    visualization_msgs::Marker key_odometrys;
    key_odometrys.header = pose_msg->header;
    key_odometrys.header.frame_id = "world";
    key_odometrys.ns = "key_odometrys";
    key_odometrys.type = visualization_msgs::Marker::SPHERE_LIST;
    key_odometrys.action = visualization_msgs::Marker::ADD;
    key_odometrys.pose.orientation.w = 1.0;
    key_odometrys.lifetime = ros::Duration();

    //static int key_odometrys_id = 0;
    key_odometrys.id = 0; //key_odometrys_id++;
    key_odometrys.scale.x = 0.1;
    key_odometrys.scale.y = 0.1;
    key_odometrys.scale.z = 0.1;
    key_odometrys.color.r = 1.0;
    key_odometrys.color.a = 1.0;
    //为了实现遍历功能，odometry_buf的元素先弹出，再压进去
    for (unsigned int i = 0; i < odometry_buf.size(); i++)
    {
        geometry_msgs::Point pose_marker;
        Vector3d vio_t;
        vio_t = odometry_buf.front();
        odometry_buf.pop();
        pose_marker.x = vio_t.x();
        pose_marker.y = vio_t.y();
        pose_marker.z = vio_t.z();
        key_odometrys.points.push_back(pose_marker);
        odometry_buf.push(vio_t);
    }
    //滑窗内相机轨迹发布出去,待订阅后在RVIZ显示，即系统运行时框框后的尾巴
    pub_key_odometrys.publish(key_odometrys);

    if (!LOOP_CLOSURE)
    {
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header = pose_msg->header;
        pose_stamped.header.frame_id = "world";
        pose_stamped.pose.position.x = vio_t.x();
        pose_stamped.pose.position.y = vio_t.y();
        pose_stamped.pose.position.z = vio_t.z();
        no_loop_path.header = pose_msg->header;
        no_loop_path.header.frame_id = "world";
        no_loop_path.poses.push_back(pose_stamped);
        pub_vio_path.publish(no_loop_path);
    }
}

//相机IMU的外参回调函数，得到相机到IMU的外参tic和qic
void extrinsic_callback(const nav_msgs::Odometry::ConstPtr &pose_msg)
{
    m_process.lock();
    tic = Vector3d(pose_msg->pose.pose.position.x,
                   pose_msg->pose.pose.position.y,
                   pose_msg->pose.pose.position.z);
    qic = Quaterniond(pose_msg->pose.pose.orientation.w,
                      pose_msg->pose.pose.orientation.x,
                      pose_msg->pose.pose.orientation.y,
                      pose_msg->pose.pose.orientation.z).toRotationMatrix();
    m_process.unlock();
}


// process中的核心操作，就是根据estimator节点当中发送的关键帧位姿来创建“符合条件”的新关键帧添加到位姿图当中
// 所有的闭环检测和重定位等操作都在posegraph.addKeyFrame当中进行
void process()
{
    if (!LOOP_CLOSURE)
        return;
    while (true)
    {
        // 三个参数图像、点云、VIO位姿
        sensor_msgs::ImageConstPtr image_msg = NULL;
        sensor_msgs::PointCloudConstPtr point_msg = NULL; //世界坐标系下的地图点云
        nav_msgs::Odometry::ConstPtr pose_msg = NULL;

        // find out the messages with same time stamp
        // 得到具有相同时间戳的pose_msg、image_msg、point_msg
        m_buf.lock();
        if(!image_buf.empty() && !point_buf.empty() && !pose_buf.empty())
        {
            //图像时间戳晚于位姿时间戳，则将该位姿pop出去
            if (image_buf.front()->header.stamp.toSec() > pose_buf.front()->header.stamp.toSec())
            {
                pose_buf.pop();
                printf("throw pose at beginning\n");
            }
                //图像时间戳晚于点云时间戳，则将点云位姿pop出去
            else if (image_buf.front()->header.stamp.toSec() > point_buf.front()->header.stamp.toSec())
            {
                point_buf.pop();
                printf("throw point at beginning\n");
            }
            else if (image_buf.back()->header.stamp.toSec() >= pose_buf.front()->header.stamp.toSec() 
                && point_buf.back()->header.stamp.toSec() >= pose_buf.front()->header.stamp.toSec())
            {
                //则pose_msg中存放的是关键帧位姿
                pose_msg = pose_buf.front();
                pose_buf.pop();
                while (!pose_buf.empty())
                    pose_buf.pop();
                while (image_buf.front()->header.stamp.toSec() < pose_msg->header.stamp.toSec())
                    image_buf.pop();
                image_msg = image_buf.front();
                image_buf.pop();

                while (point_buf.front()->header.stamp.toSec() < pose_msg->header.stamp.toSec())
                    point_buf.pop();
                point_msg = point_buf.front();
                point_buf.pop();
            }
        }
        m_buf.unlock();

        if (pose_msg != NULL)
        {
            //printf(" pose time %f \n", pose_msg->header.stamp.toSec());
            //printf(" point time %f \n", point_msg->header.stamp.toSec());
            //printf(" image time %f \n", image_msg->header.stamp.toSec());
            // skip fisrt few
            //3.剔除掉了前SKIP_FIRST_CNT（值为10）帧数据
            if (skip_first_cnt < SKIP_FIRST_CNT)
            {
                skip_first_cnt++;
                continue;
            }

            if (skip_cnt < SKIP_CNT) //SKIP_CNT在euroc.launch中为0
            {
                skip_cnt++;
                continue;
            }
            else
            {
                skip_cnt = 0;
            }
            //4.解析image_msg信息存入ptr变量当中
            //通过桥梁指针ptr将ROS图像消息转为OpenCV库可以识别的图像矩阵image
            cv_bridge::CvImageConstPtr ptr;
            if (image_msg->encoding == "8UC1")
            {
                sensor_msgs::Image img;
                img.header = image_msg->header;
                img.height = image_msg->height;
                img.width = image_msg->width;
                img.is_bigendian = image_msg->is_bigendian;
                img.step = image_msg->step;
                img.data = image_msg->data;
                img.encoding = "mono8";
                ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
            }
            else
                ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::MONO8);
            
            cv::Mat image = ptr->image;
            // build keyframe
            //5.将当前图像位置和上次图像的位置之间的距离大于SKIP_DIS的图像，创建为关键帧并加入到位姿图当中
            Vector3d T = Vector3d(pose_msg->pose.pose.position.x,
                                  pose_msg->pose.pose.position.y,
                                  pose_msg->pose.pose.position.z);
            Matrix3d R = Quaterniond(pose_msg->pose.pose.orientation.w,
                                     pose_msg->pose.pose.orientation.x,
                                     pose_msg->pose.pose.orientation.y,
                                     pose_msg->pose.pose.orientation.z).toRotationMatrix();
            //5.将当前图像位置和上次图像的位置之间的距离大于SKIP_DIS的图像，创建为关键帧并加入到位姿图当中
            if((T - last_t).norm() > SKIP_DIS)  //只用到平移向量  SKIP_DIS值为0
            {
                vector<cv::Point3f> point_3d; 
                vector<cv::Point2f> point_2d_uv; 
                vector<cv::Point2f> point_2d_normal;
                vector<double> point_id;
                //遍历点云消息中的点
                for (unsigned int i = 0; i < point_msg->points.size(); i++)
                {
                    cv::Point3f p_3d;
                    p_3d.x = point_msg->points[i].x;
                    p_3d.y = point_msg->points[i].y;
                    p_3d.z = point_msg->points[i].z;
                    point_3d.push_back(p_3d);

                    cv::Point2f p_2d_uv, p_2d_normal;
                    double p_id;
                    p_2d_normal.x = point_msg->channels[i].values[0];
                    p_2d_normal.y = point_msg->channels[i].values[1];
                    p_2d_uv.x = point_msg->channels[i].values[2];
                    p_2d_uv.y = point_msg->channels[i].values[3];
                    p_id = point_msg->channels[i].values[4];
                    point_2d_normal.push_back(p_2d_normal);
                    point_2d_uv.push_back(p_2d_uv);
                    point_id.push_back(p_id);

                    //printf("u %f, v %f \n", p_2d_uv.x, p_2d_uv.y);
                }
                //创建关键帧
                KeyFrame* keyframe = new KeyFrame(pose_msg->header.stamp.toSec(), frame_index, T, R, image,
                                   point_3d, point_2d_uv, point_2d_normal, point_id, sequence);   
                m_process.lock(); // 线程锁住
                start_flag = 1;
                //位姿图中加入关键帧，flag_detect_loop设置为1
                posegraph.addKeyFrame(keyframe, 1);
                m_process.unlock();
                frame_index++;
                last_t = T;
            }
        }
        //6.线程休眠5ms
        std::chrono::milliseconds dura(5);
        std::this_thread::sleep_for(dura);
    }
}

//command线程中要处理的工作相对要少的多，主要是检测用户键盘输入是否为's'和'n'\
// 如果接收到了键盘输入的's'，则保存位姿图到指定的路径中；如果接收到的键盘输入为'n'
// 则创建新的位姿图序列，这里最多支持5个位姿图序列,每一个关键帧都有其所在的序列（对应KeyFrame类当中的int sequence）
void command()
{
    if (!LOOP_CLOSURE)
        return;
    while(1)
    {
        char c = getchar();
        if (c == 's')
        {
            m_process.lock();
            //用户键盘输入s后，保存当前的位姿图(地图)
            posegraph.savePoseGraph();
            m_process.unlock();
            printf("save pose graph finish\nyou can set 'load_previous_pose_graph' to 1 in the config file to reuse it next time\n");
            // printf("program shutting down...\n");
            // ros::shutdown();
        }
        //检查用户键盘输入是否为n，为n则开始一个新的图像序列
        if (c == 'n')
            new_sequence();

        std::chrono::milliseconds dura(5);
        std::this_thread::sleep_for(dura);
    }
}


// 订阅的topic回调函数中对msg的解析处理基本上都是将解析出来的数据存入pose_graph节点中的buf当中
int main(int argc, char **argv)
{
    // 1.ROS初始化，设置句柄
    ros::init(argc, argv, "pose_graph");
    ros::NodeHandle n("~");
    //2.注册topic的发布
    posegraph.registerPub(n);

    // read param
    //3.读取相关参数
    //VISUALIZATION_SHIFT_X、VISUALIZATION_SHIFT_Y为可视化界面中图像x轴y轴的偏移量，一般设置为0；
    //SKIP_CNT为之后运行process()内循环的间隔；
    //SKIP_DIS为判断是否构建关键帧的距离标准；
    //visualize_camera_size为可视化界面图像的尺寸；
    n.getParam("visualization_shift_x", VISUALIZATION_SHIFT_X);
    n.getParam("visualization_shift_y", VISUALIZATION_SHIFT_Y);
    n.getParam("skip_cnt", SKIP_CNT);
    n.getParam("skip_dis", SKIP_DIS);
    std::string config_file;
    n.getParam("config_file", config_file);
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }

    double camera_visual_size = fsSettings["visualize_camera_size"];
    cameraposevisual.setScale(camera_visual_size);
    cameraposevisual.setLineWidth(camera_visual_size / 10.0);
    //配置文件中该值为1
    LOOP_CLOSURE = fsSettings["loop_closure"];
    std::string IMAGE_TOPIC;
    int LOAD_PREVIOUS_POSE_GRAPH;
    if (LOOP_CLOSURE)
    {
        //闭环情况下相关参数的读取
        ROW = fsSettings["image_height"]; //行数480
        COL = fsSettings["image_width"];  //列数752
        //读取字典
        //读取pose_graph包的地址
        std::string pkg_path = ros::package::getPath("pose_graph");
        string vocabulary_file = pkg_path + "/../support_files/brief_k10L6.bin";
        cout << "vocabulary_file" << vocabulary_file << endl;
        posegraph.loadVocabulary(vocabulary_file);
        //读取BRIEF描述子的模板文件
        BRIEF_PATTERN_FILE = pkg_path + "/../support_files/brief_pattern.yml";
        cout << "BRIEF_PATTERN_FILE" << BRIEF_PATTERN_FILE << endl;
        m_camera = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(config_file.c_str());

        fsSettings["image_topic"] >> IMAGE_TOPIC;        
        fsSettings["pose_graph_save_path"] >> POSE_GRAPH_SAVE_PATH;
        fsSettings["output_path"] >> VINS_RESULT_PATH;
        fsSettings["save_image"] >> DEBUG_IMAGE;

        // create folder if not exists
        FileSystemHelper::createDirectoryIfNotExists(POSE_GRAPH_SAVE_PATH.c_str());
        FileSystemHelper::createDirectoryIfNotExists(VINS_RESULT_PATH.c_str());

        VISUALIZE_IMU_FORWARD = fsSettings["visualize_imu_forward"];
        LOAD_PREVIOUS_POSE_GRAPH = fsSettings["load_previous_pose_graph"];
        FAST_RELOCALIZATION = fsSettings["fast_relocalization"];
        VINS_RESULT_PATH = VINS_RESULT_PATH + "/vins_result_loop.csv";
        std::ofstream fout(VINS_RESULT_PATH, std::ios::out);
        fout.close();
        fsSettings.release();
        //4.加载先前保存的位姿图
        if (LOAD_PREVIOUS_POSE_GRAPH)
        {
            printf("load pose graph\n");
            m_process.lock();
            posegraph.loadPoseGraph();
            m_process.unlock();
            printf("load pose graph finish\n");
            load_flag = 1;
        }
        else
        {
            printf("no previous pose graph\n");
            load_flag = 1;
        }
    }

    fsSettings.release();
    //5.订阅了七个topic
    ros::Subscriber sub_imu_forward = n.subscribe("/vins_estimator/imu_propagate", 2000, imu_forward_callback);
    ros::Subscriber sub_vio = n.subscribe("/vins_estimator/odometry", 2000, vio_callback);
    //订阅原图像
    ros::Subscriber sub_image = n.subscribe(IMAGE_TOPIC, 2000, image_callback);
    //订阅keyframe pose
    ros::Subscriber sub_pose = n.subscribe("/vins_estimator/keyframe_pose", 2000, pose_callback);
    //订阅相机到IMU之间的外参
    ros::Subscriber sub_extrinsic = n.subscribe("/vins_estimator/extrinsic", 2000, extrinsic_callback);
    //订阅关键帧点云topic
    ros::Subscriber sub_point = n.subscribe("/vins_estimator/keyframe_point", 2000, point_callback);
    //订阅重定位帧相对位姿
    ros::Subscriber sub_relo_relative_pose = n.subscribe("/vins_estimator/relo_relative_pose", 2000, relo_relative_pose_callback);
    //6.创建pose_graph中要发布的5个topic的发布器
    pub_match_img = n.advertise<sensor_msgs::Image>("match_image", 1000);
    pub_camera_pose_visual = n.advertise<visualization_msgs::MarkerArray>("camera_pose_visual", 1000);
    pub_key_odometrys = n.advertise<visualization_msgs::Marker>("key_odometrys", 1000);
    pub_vio_path = n.advertise<nav_msgs::Path>("no_loop_path", 1000);
    pub_match_points = n.advertise<sensor_msgs::PointCloud>("match_points", 100);

    std::thread measurement_process;
    std::thread keyboard_command_process;
    //7.创建process线程，也相当于是pose_graph的主线程
    measurement_process = std::thread(process);
    //8.创建command线程，监听命令行中键盘的输入
    keyboard_command_process = std::thread(command);

    ros::spin();

    return 0;
}
