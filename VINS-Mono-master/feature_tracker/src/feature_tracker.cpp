#include "feature_tracker.h"

int FeatureTracker::n_id = 0;

// 判断跟踪的特征点是否在图像边界内
bool inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);  //cvRound():返回跟参数最接近的整数值，即四舍五入
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}
// 去除无法跟踪的特征点,status为0的点直接跳过，否则v[j++]=v[i]留下来，最后v.resize(j)根据最新的j安排内存
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);  //重新设置v大小为j
}
// 用于track_cnt,ids更新
void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

FeatureTracker::FeatureTracker()
{
}
// 对跟踪到的特征点forw_pts，按照被跟踪次数cnt从大到小排序
// 在mask中将每个特征点周围半径为MIN_DIST的区域设置为0，后面不再选取该区域内的特征点
void FeatureTracker::setMask()
{
    if(FISHEYE)
        mask = fisheye_mask.clone();
    else
        mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255)); //CV_8UC1:1通道每个通道是无符号单字节 ////cv::Scalar(n)表示将单通道图像每个像素值初始化设为n

    // prefer to keep features that are tracked for long time
    // 构造(cnt (pts id))序列
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    for (unsigned int i = 0; i < forw_pts.size(); i++)
        // template pair make_pair(T1 a, T2 b) { return pair(a, b); }
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(forw_pts[i], ids[i])));
    //对光流跟踪到的特征点forw_pts,按照被跟踪到的次数cnt从大到小排序（lambda表达式）
    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         {
            return a.first > b.first;
         });
    //清空cnt pts id并重新存入
    forw_pts.clear();
    ids.clear();
    track_cnt.clear();

    for (auto &it : cnt_pts_id)  //引用只是给定对象的另一个名称,当我们使用引用作为我们的控制变量时,该变量依次绑定到序列中的每个元素;使用引用，我们可以更改引用所绑定的字符
    {
        if (mask.at<uchar>(it.second.first) == 255)  //mask中位置为it.second.first的值
        {
            //当前特征点位置对应的mask值为255,则保留当前特征点,将对应的特征点位置pts id 被追踪次数cnt分别存入
            forw_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            //在mask中将当前特征点周围半径为MIN_DIST的区域设置为0,后面不再选取该区域内的点（使跟踪点不集中在一个区域上）
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}
// 添加新检测到的特征点到forw_pts中
void FeatureTracker::addPoints()
{
    for (auto &p : n_pts)  //
    {
        forw_pts.push_back(p);
        ids.push_back(-1);
        track_cnt.push_back(1);
    }
}

//提取并跟踪图像特征
void FeatureTracker::readImage(const cv::Mat &_img, double _cur_time)   // _cur_time:时间戳
{
    cv::Mat img;
    TicToc t_r;
    cur_time = _cur_time;

    if (EQUALIZE)        // 如果EQUALIZE=1,表示太亮或太暗,进行直方图均衡化处理
    {
        // 自适应直方图均衡化
        // createCLAHE(对比度阈值，网格是将原图划分为8*8)
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        TicToc t_c;
        clahe->apply(_img, img); //void apply(cv::InputArray src, cv::OutputArray dst)
        ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
    }
    else
        img = _img;

    if (forw_img.empty())    //当前读入的图像是第一帧图像
    {
        prev_img = cur_img = forw_img = img;
    }
    else
    {
        forw_img = img;
    }
    //  此时forw_pts还保存的是上一帧图像中的特征点,所以把它清除
    forw_pts.clear();

    if (cur_pts.size() > 0)
    {
        TicToc t_o;
        vector<uchar> status;
        vector<float> err;
        // 对前一帧特征点cur_pts进行LK金字塔光流跟踪得到forw_pts
        // status标记了从前一帧cur_img到forw_img特征点的跟踪状态,无法被追踪到的点标记为0
        cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3);
        // 将边界外的点标记为0  //forw_pts中只含有新加入的特征点和在边界内的点
        for (int i = 0; i < int(forw_pts.size()); i++)
            if (status[i] && !inBorder(forw_pts[i]))
                status[i] = 0;
        // 根据 status剔除跟踪失败和图像边界外的点
        // 剔除时不仅要从当前帧数据forw_pts中剔除,而且还要从cur_un_pts,prev_pts,cur_pts,记录特征点id的ids,和记录特征点被跟踪次数的track_cnt中剔除
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(ids, status);
        reduceVector(cur_un_pts, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
    }
    //光流追踪成功,特征点被成功跟踪的次数就加1
    //数值代表被追踪的次数,数值越大,说明被追踪的就越久
    for (auto &n : track_cnt)
        n++;

    if (PUB_THIS_FRAME)
    {
        rejectWithF();    //通过基本矩阵剔除外点，涉及本质矩阵，基础矩阵等知识
        ROS_DEBUG("set mask begins");
        TicToc t_m;

        setMask();    //对跟踪点排序并依次选点,设置mask,去除密集点使特征点分布均匀,保证相邻的特征点之间要相隔30个像素
        ROS_DEBUG("set mask costs %fms", t_m.toc());

        ROS_DEBUG("detect feature begins");
        TicToc t_t;
        //经过外点剔除和特征点均匀化后，计算是否需要提取新的特征点
        int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
        if (n_max_cnt > 0)
        {
            if(mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            if (mask.size() != forw_img.size())
                cout << "wrong size " << endl;
            // goodFeaturesToTrack 寻找新的特征点(shi-tomasi角点),
            // 添加(MAX_CNT - forw_pts.size())个点以确保每帧都有足够的特征点
            // 在mask中不为0的区域检测新特征点
            cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - forw_pts.size(), 0.01, MIN_DIST, mask);
        }
        else
            n_pts.clear();
        ROS_DEBUG("detect feature costs: %fms", t_t.toc());

        ROS_DEBUG("add feature begins");
        TicToc t_a;
        // addPoints 向forw_pts添加新的追踪点,id初始化-1,track_cnt初始化1
        addPoints();
        ROS_DEBUG("selectFeature costs: %fms", t_a.toc());
    }
    // 当下一帧图像到来时,当前帧数据就成为了上一帧发布的数据
    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    // 把当前帧的数据forw_img forw_pts赋给上一帧cur_img cur_pts
    cur_img = forw_img;
    cur_pts = forw_pts;
    // undistortedPoints 对特征点去畸变矫正和深度归一化,计算每个角点的速度
    undistortedPoints();
    prev_time = cur_time;
}
//首先把特征点转化到归一化相机坐标系，然后计算F矩阵，再根据status清除为0的特征点
void FeatureTracker::rejectWithF()
{
    if (forw_pts.size() >= 8)
    {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_forw_pts(forw_pts.size());
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            Eigen::Vector3d tmp_p;
            //根据不同的相机模型将特征点二维图像平面坐标转化到归一化相机坐标系下三维坐标
            m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            //转换为归一化相机坐标
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
            //
            m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
        //调用cv::findFundamentalMat对un_cur_pts和un_forw_pts计算F矩阵
        // F_THRESHOLD只用于方法RANSAC或LMedS ,点到对极线的最大距离，超过这个值的点将被舍弃，不用于后面的计算,通常这个值的设定是0.5/1.0
        // param2:表示矩阵正确的可信度
        //status:具有N个元素的输出数组，在计算过程中没有被舍弃的点，元素被被置为1；否则置为0
        cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = cur_pts.size();
        // 利用status通过reduceVector去除外点
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);

        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, forw_pts.size(), 1.0 * forw_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}
//用来作为特征点id，每检测到一个新的特征点，就将n_id作为该特征点的id，然后n_id加1
bool FeatureTracker::updateID(unsigned int i)
{
    if (i < ids.size())
    {
        if (ids[i] == -1)
            ids[i] = n_id++;
        return true;
    }
    else
        return false;
}

void FeatureTracker::readIntrinsicParameter(const string &calib_file)
{
    ROS_INFO("reading paramerter of camera %s", calib_file.c_str());
    m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}
//显示去畸变的图像
void FeatureTracker::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < COL; i++)
        for (int j = 0; j < ROW; j++)  //整幅图像坐标(i,j)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + COL / 2; //pp中位置(0,0)的值
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + ROW / 2;
        pp.at<float>(2, 0) = 1.0;
        //cout << trackerData[0].K << endl;
        //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < ROW + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < COL + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    cv::imshow(name, undistortedImg);
    cv::waitKey(0);
}
//1.角点图像坐标进行去畸变矫正，转换到归一化坐标系上，2.并计算每个角点的速度
void FeatureTracker::undistortedPoints()
{
    cur_un_pts.clear();
    cur_un_pts_map.clear();
    //cv::undistortPoints(cur_pts, un_pts, K, cv::Mat());
    for (unsigned int i = 0; i < cur_pts.size(); i++)
    {
        Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y);  //所有的特征点坐标
        Eigen::Vector3d b;
        // 根据不同的相机模型将二维坐标转换到归一化相机三维坐标系
        m_camera->liftProjective(a, b);
        //再延伸到深度归一化平面上
        cur_un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
        cur_un_pts_map.insert(make_pair(ids[i], cv::Point2f(b.x() / b.z(), b.y() / b.z())));
        //printf("cur pts id %d %f %f", ids[i], cur_un_pts[i].x, cur_un_pts[i].y);
    }
    // caculate points velocity
    if (!prev_un_pts_map.empty())
    {
        double dt = cur_time - prev_time;
        pts_velocity.clear();
        for (unsigned int i = 0; i < cur_un_pts.size(); i++)
        {
            if (ids[i] != -1)  //
            {
                std::map<int, cv::Point2f>::iterator it;
                it = prev_un_pts_map.find(ids[i]);   //在上一次发布的帧中寻找相同的特征点 返回迭代器
                if (it != prev_un_pts_map.end())
                {
                    double v_x = (cur_un_pts[i].x - it->second.x) / dt;
                    double v_y = (cur_un_pts[i].y - it->second.y) / dt;
                    pts_velocity.push_back(cv::Point2f(v_x, v_y));
                }
                else
                    pts_velocity.push_back(cv::Point2f(0, 0));
            }
            else
            {
                pts_velocity.push_back(cv::Point2f(0, 0));
            }
        }
    }
    else
    {
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    prev_un_pts_map = cur_un_pts_map; //更新地图
}
