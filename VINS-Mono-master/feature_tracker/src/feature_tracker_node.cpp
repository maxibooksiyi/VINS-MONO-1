#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Bool.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>

#include "feature_tracker.h"

#define SHOW_UNDISTORTION 0

vector<uchar> r_status;
vector<float> r_err;
queue<sensor_msgs::ImageConstPtr> img_buf;

ros::Publisher pub_img,pub_match;
ros::Publisher pub_restart;

FeatureTracker trackerData[NUM_OF_CAM];
double first_image_time;
int pub_count = 1;
bool first_image_flag = true;
double last_image_time = 0;
bool init_pub = 0;

// 特征跟踪功能的入口,接收到图像以后做一些基本的处理,然后调用feature_tracker.cpp文件中关于特征跟踪功能的具体实现函数
void img_callback(const sensor_msgs::ImageConstPtr &img_msg)   //输入为一幅图像
{
    if(first_image_flag)    //是否为第一帧
    {
        first_image_flag = false;
        first_image_time = img_msg->header.stamp.toSec();
        last_image_time = img_msg->header.stamp.toSec();  //初始化last_image_time
        return;
    }
    // detect unstable camera stream
    // 时间间隔  特征间断  重新跟踪  重设第一帧图像
    if (img_msg->header.stamp.toSec() - last_image_time > 1.0 || img_msg->header.stamp.toSec() < last_image_time)
    {
        ROS_WARN("image discontinue! reset the feature tracker!");
        first_image_flag = true; 
        last_image_time = 0;
        pub_count = 1;
        std_msgs::Bool restart_flag;
        restart_flag.data = true;
        pub_restart.publish(restart_flag);   //restart_flag发布到pub_restart话题下
        return;
    }
    last_image_time = img_msg->header.stamp.toSec();
    // frequency control
    // 通过控制间隔时间内发布次数进行发布频率控制,发不发布都会执行readImage读取图像进行处理
    // FREQ:10Hz
    if (round(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time)) <= FREQ)
    {
        PUB_THIS_FRAME = true;
        // reset the frequency control
        // 时间间隔内的发布频率十分接近设定频率时，更新时间间隔起始时刻，并将数据发布次数置0
        if (abs(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time) - FREQ) < 0.01 * FREQ)
        {
            first_image_time = img_msg->header.stamp.toSec();
            pub_count = 0;
        }
    }
    else
        PUB_THIS_FRAME = false;

    cv_bridge::CvImageConstPtr ptr;  //cv_bridge将ROS的图像消息转换为OpenCV图像格式时都是通过CvImage类实现的  cv_bridge::toCvShare/toCvCopy
    if (img_msg->encoding == "8UC1")  //mono8: CV_8UC1, grayscale image
    {
        sensor_msgs::Image img;        //sensor_msgs::ImageConstPtr &img_msg
        img.header = img_msg->header;
        img.height = img_msg->height;    //rows
        img.width = img_msg->width;      //columns
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";       // 将图像编码8UC1转换为mono8
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);   //sensor_msgs是ros下的
    }
    else
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8); //当源和目的地encodings匹配,toCvCopy从ROS信息中创建图像数据的副本

    cv::Mat show_img = ptr->image;
    TicToc t_r;  //自定义时间类
    for (int i = 0; i < NUM_OF_CAM; i++)   //NUM_OF_CAM
    {
        ROS_DEBUG("processing camera %d", i);
        if (i != 1 || !STEREO_TRACK)  //单目  //STEREO_TRACK:双目跟踪则为1
            //readImage 读取图像数据进行处理
            trackerData[i].readImage(ptr->image.rowRange(ROW * i, ROW * (i + 1)), img_msg->header.stamp.toSec()); //img_msg->header.stamp.toSec():ROS时间戳
        else //双目
        {
            if (EQUALIZE) //光太亮或太暗，自适应直方图均衡化处理       //mutable关键字 修饰的数据成员，可以在 const 成员函数中修改
            {
                cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
                clahe->apply(ptr->image.rowRange(ROW * i, ROW * (i + 1)), trackerData[i].cur_img); //void apply(cv::InputArray src, cv::OutputArray dst)
            }
            else
                trackerData[i].cur_img = ptr->image.rowRange(ROW * i, ROW * (i + 1));
        }

#if SHOW_UNDISTORTION
        trackerData[i].showUndistortion("undistrotion_" + std::to_string(i));//
#endif
    }

    for (unsigned int i = 0;; i++)
    {
        bool completed = false;
        for (int j = 0; j < NUM_OF_CAM; j++)
            if (j != 1 || !STEREO_TRACK) //单目
                completed |= trackerData[j].updateID(i);  //更新全局ID 更新feature的id  |=或运算
        if (!completed)
            break;
    }
    // 进行信息封装和发布
    // 把以下信息封装成PointCloudPtr类型的feature_points实例中，发布到pub_img；
    // 将图像封装到cvtColor类型的ptr实例中发布到pub_match
   if (PUB_THIS_FRAME)
   {
        pub_count++;
        sensor_msgs::PointCloudPtr feature_points(new sensor_msgs::PointCloud);
        sensor_msgs::ChannelFloat32 id_of_point;  // id
        sensor_msgs::ChannelFloat32 u_of_point;   // 2D像素点坐标(u,v)
        sensor_msgs::ChannelFloat32 v_of_point;   //
        sensor_msgs::ChannelFloat32 velocity_x_of_point;  //2D像素点速度(vx,vy)
        sensor_msgs::ChannelFloat32 velocity_y_of_point;  //

        feature_points->header = img_msg->header;
        feature_points->header.frame_id = "world";
        // 如img_msg->channels[0].values[i]表示第i个特征点的ID值==id_of_point.value[i]
        // img_msg->channels[1].values[i]表示第i个像素点坐标的u值==u_of_point[i]
        // 以此类推....

        vector<set<int>> hash_ids(NUM_OF_CAM);
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            // auto的原理就是根据后面的值，来自己推测前面的类型是什么
            auto &un_pts = trackerData[i].cur_un_pts;     //cur_un_pts:归一化相机坐标系下的坐标
            auto &cur_pts = trackerData[i].cur_pts;
            auto &ids = trackerData[i].ids;  //
            auto &pts_velocity = trackerData[i].pts_velocity;  //pts_velocity:当前帧相对前一帧特征点沿x,y方向的像素移动速度
            for (unsigned int j = 0; j < ids.size(); j++)
            {
                if (trackerData[i].track_cnt[j] > 1)    //特征点ids[j]被追踪次数大于1
                {
                    int p_id = ids[j];   //特征点ids[j]
                    hash_id s[i].insert(p_id);
                    geometry_msgs::Point32 p;
                    p.x = un_pts[j].x;   //特征点ids[j]在归一化相机坐标系下的坐标
                    p.y = un_pts[j].y;
                    p.z = 1;

                    feature_points->points.push_back(p);    //矫正后归一化平面的3D点(x,y,z=1)
                    id_of_point.values.push_back(p_id * NUM_OF_CAM + i);
                    u_of_point.values.push_back(cur_pts[j].x);
                    v_of_point.values.push_back(cur_pts[j].y);
                    velocity_x_of_point.values.push_back(pts_velocity[j].x);
                    velocity_y_of_point.values.push_back(pts_velocity[j].y);
                }
            }
        }
        feature_points->channels.push_back(id_of_point);
        feature_points->channels.push_back(u_of_point);
        feature_points->channels.push_back(v_of_point);
        feature_points->channels.push_back(velocity_x_of_point);
        feature_points->channels.push_back(velocity_y_of_point);
        ROS_DEBUG("publish %f, at %f", feature_points->header.stamp.toSec(), ros::Time::now().toSec());

        // skip the first image; since no optical speed on frist image
        if (!init_pub)
        {
            init_pub = 1;
        }
        else
            pub_img.publish(feature_points);  //feature_points集合发布到pub_img话题下

        if (SHOW_TRACK)  //显示
        {
            // 将图像封装到cv_bridge::cvtColor类型的ptr实例中发布到pub_match
            ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::BGR8);
            //cv::Mat stereo_img(ROW * NUM_OF_CAM, COL, CV_8UC3);
            cv::Mat stereo_img = ptr->image;

            for (int i = 0; i < NUM_OF_CAM; i++)
            {
                cv::Mat tmp_img = stereo_img.rowRange(i * ROW, (i + 1) * ROW);
                cv::cvtColor(show_img, tmp_img, CV_GRAY2RGB);  //show_img灰度图转RGB（tmp_img）

                for (unsigned int j = 0; j < trackerData[i].cur_pts.size(); j++)
                {
                    double len = std::min(1.0, 1.0 * trackerData[i].track_cnt[j] / WINDOW_SIZE);
                    // 显示追踪状态，越红越好，越蓝越不行
                    cv::circle(tmp_img, trackerData[i].cur_pts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
                    //draw speed line
                    /*
                    Vector2d tmp_cur_un_pts (trackerData[i].cur_un_pts[j].x, trackerData[i].cur_un_pts[j].y);
                    Vector2d tmp_pts_velocity (trackerData[i].pts_velocity[j].x, trackerData[i].pts_velocity[j].y);
                    Vector3d tmp_prev_un_pts;
                    tmp_prev_un_pts.head(2) = tmp_cur_un_pts - 0.10 * tmp_pts_velocity;
                    tmp_prev_un_pts.z() = 1;
                    Vector2d tmp_prev_uv;
                    trackerData[i].m_camera->spaceToPlane(tmp_prev_un_pts, tmp_prev_uv);
                    cv::line(tmp_img, trackerData[i].cur_pts[j], cv::Point2f(tmp_prev_uv.x(), tmp_prev_uv.y()), cv::Scalar(255 , 0, 0), 1 , 8, 0);
                    */
                    //char name[10];
                    //sprintf(name, "%d", trackerData[i].ids[j]);
                    //cv::putText(tmp_img, name, trackerData[i].cur_pts[j], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
                }
            }
            //cv::imshow("vis", stereo_img);
            //cv::waitKey(5);
            pub_match.publish(ptr->toImageMsg());  //图像发布到pub_match话题下
        }
    }
    ROS_INFO("whole feature tracker processing costs: %f", t_r.toc());
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "feature_tracker");
    ros::NodeHandle n("~");  // 命名空间为/node_namespace/node_name
    //设置logger的级别,只有级别大于或等于level的日志记录消息才会得到处理
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    // 1.读取yaml中的一些配置参数
    readParameters(n);      //通过ROS来读参数

    // 2.读取每个相机实例对应的相机内参，NUM_OF_CAM 经常为1，单目
    for (int i = 0; i < NUM_OF_CAM; i++)  //NUM_OF_CAM=1为单目
        trackerData[i].readIntrinsicParameter(CAM_NAMES[i]);    //读取每个相机内参

    if(FISHEYE)      //是否加入鱼眼mask来去除边缘噪声
    {
        for (int i = 0; i < NUM_OF_CAM; i++)   //NUM_OF_CAM
        {
            trackerData[i].fisheye_mask = cv::imread(FISHEYE_MASK, 0);
            if(!trackerData[i].fisheye_mask.data)
            {
                ROS_INFO("load mask fail");
                ROS_BREAK();
            }
            else
                ROS_INFO("load mask success");
        }
    }
    // 订阅IMAGE_TOPIC,执行img_callback
    // subscribe("topic",queue_size,address of callback function)
    //第二个参数是队列大小，当缓存达到 100 条消息后，自动舍弃时间戳最早的消息
    ros::Subscriber sub_img = n.subscribe(IMAGE_TOPIC, 100, img_callback);
    // 发布topic
    // feature：跟踪的特征点用于后端优化,由/vins_estimator订阅并进行优化
    pub_img = n.advertise<sensor_msgs::PointCloud>("feature", 1000);
    // feature_img：跟踪的特征点图,用于RVIZ和调试
    pub_match = n.advertise<sensor_msgs::Image>("feature_img",1000);
    // 判断特征跟踪模块是否出错,若有问题则进行复位,由/vins_estimator订阅
    pub_restart = n.advertise<std_msgs::Bool>("restart",1000);
    /*
    if (SHOW_TRACK)
        cv::namedWindow("vis", cv::WINDOW_NORMAL);
    */
    /*ros::spin() 进入自循环，可以尽可能快的调用消息回调函数，会调用主程序中所有回调函数，此处只有img_callback()
    一旦 ros::ok() 返回 false，ros::spin() 就会立刻跳出自循环
    这有可能是 ros::shutdown() 被调用，或者是用户按下了 Ctrl-C，使得 master 告诉节点要终止运行
    */
    ros::spin();
    return 0;
}


// new points velocity is 0, pub or not?
// track cnt > 1 pub?