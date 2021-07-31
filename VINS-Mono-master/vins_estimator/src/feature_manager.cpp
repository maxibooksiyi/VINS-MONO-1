#include "feature_manager.h"

int FeaturePerId::endFrame()
{
    return start_frame + feature_per_frame.size() - 1;
}

FeatureManager::FeatureManager(Matrix3d _Rs[])
    : Rs(_Rs)
{
    for (int i = 0; i < NUM_OF_CAM; i++)
        ric[i].setIdentity();
}

void FeatureManager::setRic(Matrix3d _ric[])
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ric[i] = _ric[i];
    }
}

void FeatureManager::clearState()
{
    feature.clear();
}
//该特征点被观测到两次以上,且第一次被观测到的图像的帧号不是最后一帧
int FeatureManager::getFeatureCount()
{
    int cnt = 0;
    for (auto &it : feature)
    {
        it.used_num = it.feature_per_frame.size();  //所有特征点被观测到的帧数
        if (it.used_num >= 2 && it.start_frame < WINDOW_SIZE - 2)
        {
            cnt++;  //这个特征点是有效的
        }
    }
    return cnt;
}

// 为了维护窗口的大小(局部优化关键帧的数目大小)，比较次新帧和次次新帧的视差大小，决定删除最旧帧还是次新帧
// 依据两帧之间的视差和特征点的数量决定次新帧是否作为关键帧
// 先把特征点从image放入feature的list容器中，计算每一个点跟踪次数小于阈值 和 它在次新帧和次次新帧间所有特征点的平均视差大于阈值，返回是否是关键帧
bool FeatureManager::addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td)
{
    ROS_DEBUG("input feature: %d", (int)image.size());
    ROS_DEBUG("num of feature: %d", getFeatureCount());
    double parallax_sum = 0;  //所有特征点视差总和
    int parallax_num = 0;
    last_track_num = 0;
    // 1. 把image map中的所有特征点放入feature list容器中
    // 遍历特征点,看该特征点是否在特征点的列表中,如果没在,则将<FeatureID,Start_frame>存入到Feature列表中；否则统计数目
    for (auto &id_pts : image)  //当前帧的所有特征点
    {
        FeaturePerFrame f_per_fra(id_pts.second[0].second, td);

        int feature_id = id_pts.first;
        // 在前一帧中寻找是否有该特征点
        auto it = find_if(feature.begin(), feature.end(), [feature_id](const FeaturePerId &it)
                          { return it.feature_id == feature_id; });
        // 若没有找到,则加入到特征集中
        if (it == feature.end())
        {
            feature.push_back(FeaturePerId(feature_id, frame_count)); //frame_count:当前帧数
            feature.back().feature_per_frame.push_back(f_per_fra);
        }
        else if (it->feature_id == feature_id)
        {
            it->feature_per_frame.push_back(f_per_fra);
            last_track_num++;   //此帧有多少相同特征点被跟踪
        }
    }
    // 2. 该帧中与前一帧中相同特征点数目小于20或者窗口内帧的数目小于2，是关键帧
     //frame_count<2,也就是说加入到窗口中的帧个数取值为0或者1的时候
     //last_track_num说明能够跟踪到的特征点数量少于20个
    if (frame_count < 2 || last_track_num < 20)
        return true;
    // 3.计算每个特征点在次新帧和次次新帧中的视差
    for (auto &it_per_id : feature)
    {    // 观测特征点的起始帧小于倒数第三帧,终止帧要大于倒数第二帧,保证至少有两帧能观测到
         //计算能被当前帧和其前两帧共同看到的特征点视差
         //it_per_id.feature_per_frame.size()表示该特征点能够被多少帧共视
        if (it_per_id.start_frame <= frame_count - 2 &&
            it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >= frame_count - 1)  // int(it_per_id.feature_per_frame.size()) 必须大于2
        {
            //总视差:特征点在两帧的归一化平面上的坐标点的距离
            //计算特征点it_per_id在倒数第二帧和倒数第三帧之间的视差，并求所有视差的累加和
            parallax_sum += compensatedParallax2(it_per_id, frame_count);
            parallax_num++;   //个数
        }
    }
    //4.第一次加进去的，是关键帧
    if (parallax_num == 0)
    {
        return true;
    }
    else
    {
        ROS_DEBUG("parallax_sum: %lf, parallax_num: %d", parallax_sum, parallax_num);
        ROS_DEBUG("current parallax: %lf", parallax_sum / parallax_num * FOCAL_LENGTH);
        // 平均视差大于阈值的是关键帧
        // MIN_PARALLAX=10.0/460.0
        return parallax_sum / parallax_num >= MIN_PARALLAX;
    }
}

void FeatureManager::debugShow()
{
    ROS_DEBUG("debug show");
    for (auto &it : feature)
    {
        ROS_ASSERT(it.feature_per_frame.size() != 0);
        ROS_ASSERT(it.start_frame >= 0);
        ROS_ASSERT(it.used_num >= 0);

        ROS_DEBUG("%d,%d,%d ", it.feature_id, it.used_num, it.start_frame);
        int sum = 0;
        for (auto &j : it.feature_per_frame)
        {
            ROS_DEBUG("%d,", int(j.is_used));
            sum += j.is_used;
            printf("(%lf,%lf) ",j.point(0), j.point(1));
        }
        ROS_ASSERT(it.used_num == sum);
    }
}

// 得到给定两帧之间的对应特征点的3D坐标
vector<pair<Vector3d, Vector3d>> FeatureManager::getCorresponding(int frame_count_l, int frame_count_r)
{
    vector<pair<Vector3d, Vector3d>> corres;
    for (auto &it : feature)  //先找到对应的特征点
    {
        // 要找特征点的两帧在窗口范围内,可以直接取  窗口为：观测到当前特征点的所有图像帧
        if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r)
        {
            Vector3d a = Vector3d::Zero(), b = Vector3d::Zero();
            int idx_l = frame_count_l - it.start_frame;
            int idx_r = frame_count_r - it.start_frame;

            a = it.feature_per_frame[idx_l].point;

            b = it.feature_per_frame[idx_r].point;
            
            corres.push_back(make_pair(a, b));
        }
    }
    return corres;
}

//设置特征点的逆深度估计值
void FeatureManager::setDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();  //能够观测到某个特征点的所有相关帧数目
        // 至少两帧观测得到这个特征点 且 首次观测到该特征点的图像帧在滑动窗范围内
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        // 求解逆深度
        it_per_id.estimated_depth = 1.0 / x(++feature_index); // 1.0/x(0) ?
        //ROS_INFO("feature id %d , start_frame %d, depth %f ", it_per_id->feature_id, it_per_id-> start_frame, it_per_id->estimated_depth);
        if (it_per_id.estimated_depth < 0)
        {
            it_per_id.solve_flag = 2;  //失败估计
        }
        else
            it_per_id.solve_flag = 1;  //成功估计
    }
}

void FeatureManager::removeFailures()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        if (it->solve_flag == 2)
            feature.erase(it);
    }
}

void FeatureManager::clearDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth = 1.0 / x(++feature_index);
    }
}

VectorXd FeatureManager::getDepthVector()
{
    VectorXd dep_vec(getFeatureCount());
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
#if 1
        dep_vec(++feature_index) = 1. / it_per_id.estimated_depth;  //逆深度
#else
        dep_vec(++feature_index) = it_per_id->estimated_depth;
#endif
    }
    return dep_vec;
}

// 对特征点进行三角化求深度(SVD分解)
// 注意:在VINS的定义里,世界坐标系指的就是窗口中第一帧图像的坐标系
// 同样的,所有存储的特征点的3D坐标都是相对这一帧来表示的,所以每当需要marginalize掉最后一帧时，
// 都要调用FeatureManager类中的removeBackShift()来把所有的特征点坐标调整一遍,让它们参考倒数第二帧来表示
void FeatureManager::triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[])
{
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        //需要至少两帧观测到该特征点且首次观测到特征点的帧不是倒数第三帧
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
       // 对滑动窗口中所有帧中depth未知的特征点进行三角化
        if (it_per_id.estimated_depth > 0)
            continue;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

        ROS_ASSERT(NUM_OF_CAM == 1);
        Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frame.size(), 4);  // x*y的维度
        int svd_idx = 0;

        Eigen::Matrix<double, 3, 4> P0;
        // Ps[i]:第i帧在世界坐标系下的位移
        // Rs[imu_i]:从第i帧的imu参考系旋转到世界参考系（窗口中第一帧图像坐标系）的旋转矩阵
        // tic[0]:imu参考系下第1个相机的位移
        // ric[0]:imu参考系下第1个相机的旋转矩阵
        Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];
        Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];
        // leftCols<3>()从左边第3列起到结尾,rightCols<1>()从右边第1列起到结尾
        P0.leftCols<3>() = Eigen::Matrix3d::Identity();
        P0.rightCols<1>() = Eigen::Vector3d::Zero();

        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            // R t为第j帧相机坐标系到第i帧相机坐标系的变换矩阵
            // P为i到j的变换矩阵  故要对R t进行处理
            Eigen::Vector3d t1 = Ps[imu_j] + Rs[imu_j] * tic[0];
            Eigen::Matrix3d R1 = Rs[imu_j] * ric[0];
            Eigen::Vector3d t = R0.transpose() * (t1 - t0);
            Eigen::Matrix3d R = R0.transpose() * R1;
            Eigen::Matrix<double, 3, 4> P;
            P.leftCols<3>() = R.transpose();
            P.rightCols<1>() = -R.transpose() * t;
            // 特征点表示在i帧中的3D坐标(x,y,1)
            Eigen::Vector3d f = it_per_frame.point.normalized();
            // 直接线性变换法 DLT
            svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
            svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);

            if (imu_i == imu_j)
                continue;
        }
        //对A的SVD分解得到其最小奇异值对应的单位奇异向量(x,y,z,w)，深度为z/w
        ROS_ASSERT(svd_idx == svd_A.rows());
        // SVD分解:奇异值从大到小排列    ComputeThinV:因为MatrixXd的列数未定
        Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        double svd_method = svd_V[2] / svd_V[3];
        //it_per_id->estimated_depth = -b / A;
        //it_per_id->estimated_depth = svd_V[2] / svd_V[3];

        it_per_id.estimated_depth = svd_method;
        //it_per_id->estimated_depth = INIT_DEPTH;

        if (it_per_id.estimated_depth < 0.1)
        {
            it_per_id.estimated_depth = INIT_DEPTH;
        }

    }
}

void FeatureManager::removeOutlier()
{
    ROS_BREAK();
    int i = -1;
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        i += it->used_num != 0;
        if (it->used_num != 0 && it->is_outlier == true)
        {
            feature.erase(it);
        }
    }
}

// 边缘化最老帧时,处理特征点保存的帧号,将起始帧是最老帧的特征点的深度值进行转移
// marg_R marg_P为被边缘化的位姿，new_R new_P为在这下一帧的位姿
void FeatureManager::removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P)
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            Eigen::Vector3d uv_i = it->feature_per_frame[0].point;
            // c.erase(p),c.erase(b,e);第一个删除迭代器p所指向的元素，第二个删除迭代器b,e所标记的范围内的元素，c为容器对象
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() < 2)
            {
                // 特征点只在最老帧被观测则直接移除
                feature.erase(it);
                continue;
            }
            else
            {
                //pts_i为特征点在最老帧坐标系下的三维坐标
                //w_pts_i为特征点在世界坐标系下的三维坐标
                //将其转换到在下一帧坐标系下的坐标pts_j
                Eigen::Vector3d pts_i = uv_i * it->estimated_depth;
                Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;
                Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P);
                double dep_j = pts_j(2);
                if (dep_j > 0)
                    it->estimated_depth = dep_j;
                else
                    it->estimated_depth = INIT_DEPTH;
            }
        }
        // remove tracking-lost feature after marginalize
        /*
        if (it->endFrame() < WINDOW_SIZE - 1)
        {
            feature.erase(it);
        }
        */
    }
}
// 边缘化最老帧时,直接将特征点所保存的帧号向前滑动
void FeatureManager::removeBack()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}

// 边缘化次新帧时,对特征点在次新帧的信息进行移除处理
void FeatureManager::removeFront(int frame_count)
{
    for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next)
    {
        it_next++;
        //起始帧为最新帧的滑动成次新帧
        if (it->start_frame == frame_count)
        {
            it->start_frame--;
        }
        else
        {
            int j = WINDOW_SIZE - 1 - it->start_frame;
            //如果次新帧之前已经跟踪结束则什么都不做
            if (it->endFrame() < frame_count - 1)
                continue;
            //如果在次新帧仍被跟踪，则删除feature_per_frame中次新帧对应的FeaturePerFrame
            it->feature_per_frame.erase(it->feature_per_frame.begin() + j);
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}
//计算某个特征点it_per_id在次新帧和次次新帧的视差
// 判断观测到该特征点的frame中倒数第二帧和倒数第三帧的共视关系 实际是求取该特征点在两帧的归一化平面上的坐标点的距离
double FeatureManager::compensatedParallax2(const FeaturePerId &it_per_id, int frame_count)
{
    //check the second last frame is keyframe or not
    //parallax betwwen seconde last frame and third last frame  ?
    const FeaturePerFrame &frame_i = it_per_id.feature_per_frame[frame_count - 2 - it_per_id.start_frame];
    const FeaturePerFrame &frame_j = it_per_id.feature_per_frame[frame_count - 1 - it_per_id.start_frame];

    double ans = 0;
    Vector3d p_j = frame_j.point;  // 3D路标点(倒数第二帧j)

    double u_j = p_j(0);
    double v_j = p_j(1);

    Vector3d p_i = frame_i.point;  // 3D路标点(倒数第三帧i)
    Vector3d p_i_comp;

    //int r_i = frame_count - 2;
    //int r_j = frame_count - 1;
    //p_i_comp = ric[camera_id_j].transpose() * Rs[r_j].transpose() * Rs[r_i] * ric[camera_id_i] * p_i;
    p_i_comp = p_i;
    double dep_i = p_i(2);
    double u_i = p_i(0) / dep_i;
    double v_i = p_i(1) / dep_i;
    double du = u_i - u_j, dv = v_i - v_j;

    double dep_i_comp = p_i_comp(2);
    double u_i_comp = p_i_comp(0) / dep_i_comp;
    double v_i_comp = p_i_comp(1) / dep_i_comp;
    double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;
    // 计算视差(斜边)
    ans = max(ans, sqrt(min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));

    return ans;
}