#include "pose_graph.h"

// 创建了一个优化4自由度位姿的线程，全局优化就是在这个线程t_optimization当中进行的
PoseGraph::PoseGraph()
{
    posegraph_visualization = new CameraPoseVisualization(1.0, 0.0, 1.0, 1.0);
    posegraph_visualization->setScale(0.1);
    posegraph_visualization->setLineWidth(0.01);
	t_optimization = std::thread(&PoseGraph::optimize4DoF, this);     //创建位姿优化线程
    earliest_loop_index = -1;
    t_drift = Eigen::Vector3d(0, 0, 0);
    yaw_drift = 0;
    r_drift = Eigen::Matrix3d::Identity();
    w_t_vio = Eigen::Vector3d(0, 0, 0);
    w_r_vio = Eigen::Matrix3d::Identity();
    global_index = 0;
    sequence_cnt = 0;
    sequence_loop.push_back(0);
    base_sequence = 1;

}

PoseGraph::~PoseGraph()
{
	t_optimization.join();
}

void PoseGraph::registerPub(ros::NodeHandle &n)
{
    pub_pg_path = n.advertise<nav_msgs::Path>("pose_graph_path", 1000);
    pub_base_path = n.advertise<nav_msgs::Path>("base_path", 1000);
    pub_pose_graph = n.advertise<visualization_msgs::MarkerArray>("pose_graph", 1000);
    for (int i = 1; i < 10; i++)
        pub_path[i] = n.advertise<nav_msgs::Path>("path_" + to_string(i), 1000);
}

void PoseGraph::loadVocabulary(std::string voc_path)
{
    voc = new BriefVocabulary(voc_path);
    db.setVocabulary(*voc, false, 0);
}


// 添加关键帧
// 1）对当前关键帧进行了位姿更新；
// 2）需要进行闭环检测的情况下调用detectLoop函数进行闭环检测，不需要进行闭环检测时将当前关键帧加入到词袋当中；
// 3）如果需要进行闭环检测，并且检测到了闭环的存在，则根据检测到的闭环帧的index获取该闭环帧对象，并进一步判断当前关键帧和闭环候选帧之间的闭环情况（调用findConnection函数）。
// 4）如果当前关键帧和闭环候选帧满足闭环条件（在findConnection函数中判断满足条件时返回true）,则利用旧的闭环帧位姿来计算当前帧在世界坐标系下的位姿。
// 5）计算当前帧在世界坐标系下的位姿和当前帧的位姿偏差
// 6）若检测到的闭环帧和当前帧不在一个序列，并且sequence_loop中当前帧所处的序列对应值为0（没有调整过位姿），则更新当前帧的位姿，并将关键帧列表中和当前帧处于相同序列的其他关键帧位姿进行更新。
// 7）在6）执行完后，将更新后的当前关键帧的index插入到optimize_buf当中，这样在优化线程中就可以进行优化了
// 8）将当前关键帧的位姿插入到path当中，后续的publish函数会将相关topic发布出去
// 9）如果要保存闭环路径的话，将当前关键帧的位姿插入到VINS_RESULT_PATH文件当中
// 10）如果在posegraph图中要显示sequential edge（当前关键帧和关键帧列表中其他相同序列当中关键帧之间的边）的话，则要在posegraph图中显示
// 11）如果要显示当前关键帧和与其形成闭环的关键帧之间的连线的话，则在posegraph图中添加对应的边
// 12）将当前关键帧加入到关键帧列表当中
void PoseGraph::addKeyFrame(KeyFrame* cur_kf, bool flag_detect_loop)
{
    //shift to base frame
    Vector3d vio_P_cur;  //vio 当前帧的位姿
    Matrix3d vio_R_cur;
    if (sequence_cnt != cur_kf->sequence) // 图像序列？
    {
        sequence_cnt++;  //新建一个图像序列
        sequence_loop.push_back(0);
        w_t_vio = Eigen::Vector3d(0, 0, 0);
        w_r_vio = Eigen::Matrix3d::Identity();
        m_drift.lock();  //
        t_drift = Eigen::Vector3d(0, 0, 0);
        r_drift = Eigen::Matrix3d::Identity();
        m_drift.unlock();  //
    }
    // 获取当前帧的位姿vio_P_cur vio_R_cur并更新
    // vio_R_cur：vio中当前帧的旋转矩阵
    cur_kf->getVioPose(vio_P_cur, vio_R_cur);
    vio_P_cur = w_r_vio * vio_P_cur + w_t_vio;   //更新 两次得到的可能不一样
    vio_R_cur = w_r_vio *  vio_R_cur;
    cur_kf->updateVioPose(vio_P_cur, vio_R_cur);
    cur_kf->index = global_index;
    global_index++;

	int loop_index = -1;
    if (flag_detect_loop)
    {
        TicToc tmp_t;
        // 进行回环检测
        loop_index = detectLoop(cur_kf, cur_kf->index);
    }
    else
    {
        addKeyFrameIntoVoc(cur_kf);
    }
	if (loop_index != -1) //如果存在回环候选帧，即loop_index != -1：
	{
        //printf(" %d detect loop with %d \n", cur_kf->index, loop_index);
        //返回回环候选帧的索引
        KeyFrame* old_kf = getKeyFrame(loop_index);  //get key frame infromation
        if (cur_kf->findConnection(old_kf))  //当前帧与回环候选帧进行描述子匹配
        {
            //earliest_loop_index为最早的回环候选帧
            if (earliest_loop_index > loop_index || earliest_loop_index == -1) //
                earliest_loop_index = loop_index;
            //计算当前帧与回环帧的相对位姿，纠正当前帧位姿w_P_cur w_R_cur
            Vector3d w_P_old, w_P_cur, vio_P_cur;
            Matrix3d w_R_old, w_R_cur, vio_R_cur;
            // w_P_old: P^w1_bi   w_R_old:R^w1_bi  崔华坤论文部分
            old_kf->getVioPose(w_P_old, w_R_old);  // loop frame
            // vio_P_cur: P^w2_bj   vio_R_cur:R^w2_bj   崔华坤论文部分
            cur_kf->getVioPose(vio_P_cur, vio_R_cur);  // cur frame
            //获取当前帧与回环帧的相对位姿relative_q、relative_t
            Vector3d relative_t;
            Quaterniond relative_q;
            relative_t = cur_kf->getLoopRelativeT();
            relative_q = (cur_kf->getLoopRelativeQ()).toRotationMatrix();
            // 重新计算当前帧位姿w_P_cur w_R_cur
            // w_P_cur:当前帧在世界坐标系下的位姿
            w_P_cur = w_R_old * relative_t + w_P_old;  // w_P_cur: P^w1_bj
            w_R_cur = w_R_old * relative_q;  // w_R_cur: R^w1_bj
            //回环得到的位姿和VIO位姿之间的偏移量shift_r shift_t
            double shift_yaw;
            Matrix3d shift_r;
            Vector3d shift_t;
            //计算根据闭环帧得到的位姿和当前vio之间的偏差
            shift_yaw = Utility::R2ypr(w_R_cur).x() - Utility::R2ypr(vio_R_cur).x();
            // 旋转偏差
            shift_r = Utility::ypr2R(Vector3d(shift_yaw, 0, 0));
            // 平移的偏差
            shift_t = w_P_cur - w_R_cur * vio_R_cur.transpose() * vio_P_cur; 
            // shift vio pose of whole sequence to the world frame
            // 若检测到的闭环帧和当前帧不在一个序列，并且sequence_loop中当前帧所处的序列对应值为0（没有调整过位姿）
            if (old_kf->sequence != cur_kf->sequence && sequence_loop[cur_kf->sequence] == 0)
            {  
                w_r_vio = shift_r;
                w_t_vio = shift_t;
                // w_r_vio: r^w1_w2     w_t_vio: t^w1_w2
                vio_P_cur = w_r_vio * vio_P_cur + w_t_vio;  //
                vio_R_cur = w_r_vio * vio_R_cur;     //
                //更新当前帧的位姿
                cur_kf->updateVioPose(vio_P_cur, vio_R_cur);
                list<KeyFrame*>::iterator it = keyframelist.begin();
                //遍历keyframelist链上每一个keyframe
                for (; it != keyframelist.end(); it++)   
                {
                    //遍历的keyframe和当前关键帧的序列号相同
                    if((*it)->sequence == cur_kf->sequence)
                    {
                        Vector3d vio_P_cur;
                        Matrix3d vio_R_cur;
                        //更新it的位姿
                        (*it)->getVioPose(vio_P_cur, vio_R_cur);
                        vio_P_cur = w_r_vio * vio_P_cur + w_t_vio;
                        vio_R_cur = w_r_vio *  vio_R_cur;
                        (*it)->updateVioPose(vio_P_cur, vio_R_cur);
                    }
                }
                sequence_loop[cur_kf->sequence] = 1;//
            }
            // 4.4 将当前帧放入优化队列中
            m_optimize_buf.lock();
            optimize_buf.push(cur_kf->index); //
            m_optimize_buf.unlock();
        }
	}
	m_keyframelist.lock();
    Vector3d P;
    Matrix3d R;
    // 5.获取VIO当前帧的位姿P R，根据偏移量得到实际位姿
    cur_kf->getVioPose(P, R);
    P = r_drift * P + t_drift;
    R = r_drift * R;
    cur_kf->updatePose(P, R);  // P R 为世界系w1下的
    // 6.发布path[sequence_cnt]
    Quaterniond Q{R};
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header.stamp = ros::Time(cur_kf->time_stamp);
    pose_stamped.header.frame_id = "world";
    pose_stamped.pose.position.x = P.x() + VISUALIZATION_SHIFT_X;
    pose_stamped.pose.position.y = P.y() + VISUALIZATION_SHIFT_Y;
    pose_stamped.pose.position.z = P.z();
    pose_stamped.pose.orientation.x = Q.x();
    pose_stamped.pose.orientation.y = Q.y();
    pose_stamped.pose.orientation.z = Q.z();
    pose_stamped.pose.orientation.w = Q.w();
    path[sequence_cnt].poses.push_back(pose_stamped);
    path[sequence_cnt].header = pose_stamped.header;
    // 7.保存闭环轨迹到VINS_RESULT_PATH
    if (SAVE_LOOP_PATH)
    {
        //在euroc_config.yaml当中设置的VINS_RESULT_PATH路径为output_path: "/home/shaozu/output/"
        ofstream loop_path_file(VINS_RESULT_PATH, ios::app);
        loop_path_file.setf(ios::fixed, ios::floatfield);
        loop_path_file.precision(0);
        loop_path_file << cur_kf->time_stamp * 1e9 << ",";
        loop_path_file.precision(5);
        loop_path_file  << P.x() << ","
              << P.y() << ","
              << P.z() << ","
              << Q.w() << ","
              << Q.x() << ","
              << Q.y() << ","
              << Q.z() << ","
              << endl;
        loop_path_file.close();
    }
    //draw local connection
    // SHOW_S_EDGE表示是否在位姿图中展示sequential edge，就是一个关键帧和它之前的关键帧之间的相对变换关系
    if (SHOW_S_EDGE)
    {
        list<KeyFrame*>::reverse_iterator rit = keyframelist.rbegin();
        for (int i = 0; i < 4; i++)
        {
            if (rit == keyframelist.rend())
                break;
            Vector3d conncected_P;
            Matrix3d connected_R;
            if((*rit)->sequence == cur_kf->sequence)
            {
                (*rit)->getPose(conncected_P, connected_R);
                //P是当前帧的位置，conncected_P是rit帧的位置
                posegraph_visualization->add_edge(P, conncected_P);
            }
            rit++;
        }
    }
    //当前帧与其回环帧连线
    //SHOW_L_DEBUG表示是否在位姿图中展示loop-closure edge
    if (SHOW_L_EDGE)
    {
        if (cur_kf->has_loop)
        {
            //printf("has loop \n");
            KeyFrame* connected_KF = getKeyFrame(cur_kf->loop_index);
            Vector3d connected_P,P0;
            Matrix3d connected_R,R0;
            connected_KF->getPose(connected_P, connected_R);
            //cur_kf->getVioPose(P0, R0);
            cur_kf->getPose(P0, R0);
            if(cur_kf->sequence > 0)
            {
                //printf("add loop into visual \n");
                posegraph_visualization->add_loopedge(P0, connected_P + Vector3d(VISUALIZATION_SHIFT_X, VISUALIZATION_SHIFT_Y, 0));
            }
            
        }
    }
    //posegraph_visualization->add_pose(P + Vector3d(VISUALIZATION_SHIFT_X, VISUALIZATION_SHIFT_Y, 0), Q);
    //将当前帧加入到关键帧列表当中
	keyframelist.push_back(cur_kf);
    publish();
	m_keyframelist.unlock();
}


void PoseGraph::loadKeyFrame(KeyFrame* cur_kf, bool flag_detect_loop)
{
    cur_kf->index = global_index;
    global_index++;
    int loop_index = -1;
    if (flag_detect_loop)
       loop_index = detectLoop(cur_kf, cur_kf->index);
    else
    {
        addKeyFrameIntoVoc(cur_kf);
    }
    if (loop_index != -1)
    {
        printf(" %d detect loop with %d \n", cur_kf->index, loop_index);
        KeyFrame* old_kf = getKeyFrame(loop_index);
        if (cur_kf->findConnection(old_kf))
        {
            if (earliest_loop_index > loop_index || earliest_loop_index == -1)
                earliest_loop_index = loop_index;
            m_optimize_buf.lock();
            optimize_buf.push(cur_kf->index);
            m_optimize_buf.unlock();
        }
    }
    m_keyframelist.lock();
    Vector3d P;
    Matrix3d R;
    cur_kf->getPose(P, R);
    Quaterniond Q{R};
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header.stamp = ros::Time(cur_kf->time_stamp);
    pose_stamped.header.frame_id = "world";
    pose_stamped.pose.position.x = P.x() + VISUALIZATION_SHIFT_X;
    pose_stamped.pose.position.y = P.y() + VISUALIZATION_SHIFT_Y;
    pose_stamped.pose.position.z = P.z();
    pose_stamped.pose.orientation.x = Q.x();
    pose_stamped.pose.orientation.y = Q.y();
    pose_stamped.pose.orientation.z = Q.z();
    pose_stamped.pose.orientation.w = Q.w();
    base_path.poses.push_back(pose_stamped);
    base_path.header = pose_stamped.header;

    //draw local connection
    if (SHOW_S_EDGE)
    {
        list<KeyFrame*>::reverse_iterator rit = keyframelist.rbegin();
        for (int i = 0; i < 1; i++)
        {
            if (rit == keyframelist.rend())
                break;
            Vector3d conncected_P;
            Matrix3d connected_R;
            if((*rit)->sequence == cur_kf->sequence)
            {
                (*rit)->getPose(conncected_P, connected_R);
                posegraph_visualization->add_edge(P, conncected_P);
            }
            rit++;
        }
    }
    /*
    if (cur_kf->has_loop)
    {
        KeyFrame* connected_KF = getKeyFrame(cur_kf->loop_index);
        Vector3d connected_P;
        Matrix3d connected_R;
        connected_KF->getPose(connected_P,  connected_R);
        posegraph_visualization->add_loopedge(P, connected_P, SHIFT);
    }
    */

    keyframelist.push_back(cur_kf);
    //publish();
    m_keyframelist.unlock();
}

KeyFrame* PoseGraph::getKeyFrame(int index)
{
//    unique_lock<mutex> lock(m_keyframelist);
    list<KeyFrame*>::iterator it = keyframelist.begin();
    for (; it != keyframelist.end(); it++)   
    {
        if((*it)->index == index)
            break;
    }
    if (it != keyframelist.end())
        return *it;
    else
        return NULL;
}

// detectLoop（）回环检测返回候选帧索引
int PoseGraph::detectLoop(KeyFrame* keyframe, int frame_index)
{
    // put image into image_pool; for visualization
    cv::Mat compressed_image;
    //是否将图像存入位姿图中来进行可视化处理
    if (DEBUG_IMAGE)
    {
        int feature_num = keyframe->keypoints.size();
        cv::resize(keyframe->image, compressed_image, cv::Size(376, 240));
        putText(compressed_image, "feature_num:" + to_string(feature_num), cv::Point2f(10, 10), CV_FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255));
        image_pool[frame_index] = compressed_image;
    }
    TicToc tmp_t;
    //first query; then add this frame into database!
    QueryResults ret;
    TicToc t_query;
    // 1.查询字典数据库，得到与每一帧的相似度评分ret
    /**
     在数据库(字典)中查询该关键帧的描述子，获得查询结果存入ret中
     frame_index - 50表示要查询的帧中index最大的值，说明当前帧的前50帧不在描述子匹配查询范围之内
    */
    db.query(keyframe->brief_descriptors, ret, 4, frame_index - 50);
    //printf("query time: %f", t_query.toc());
    //cout << "Searching for Image " << frame_index << ". " << ret << endl;

    TicToc t_add;
    // 2.添加当前关键帧到字典数据库中
    db.add(keyframe->brief_descriptors);
    //printf("add feature time: %f", t_add.toc());
    // ret[0] is the nearest neighbour's score. threshold change with neighour score
    // 3.通过相似度评分判断是否存在回环候选帧
    bool find_loop = false;
    cv::Mat loop_result;
    // visual loop result 视觉闭环结果
    if (DEBUG_IMAGE)
    {
        loop_result = compressed_image.clone();
        if (ret.size() > 0)
            putText(loop_result, "neighbour score:" + to_string(ret[0].Score), cv::Point2f(10, 50), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255));
    }
    // visual loop result 
    if (DEBUG_IMAGE)
    {
        for (unsigned int i = 0; i < ret.size(); i++)
        {
            int tmp_index = ret[i].Id;
            auto it = image_pool.find(tmp_index);
            cv::Mat tmp_image = (it->second).clone();
            putText(tmp_image, "index:  " + to_string(tmp_index) + "loop score:" + to_string(ret[i].Score), cv::Point2f(10, 50), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255));
            //hconcat()通过相似度评分判断是否存在回环候选帧
            //hconcat函数：用于两个Mat矩阵或者图像的水平拼接
            // hconcat（B,C，A）等同于A=[B C]
            cv::hconcat(loop_result, tmp_image, loop_result);
        }
    }
    // a good match with its nerghbour
    // 确保与相邻帧具有好的相似度评分
    if (ret.size() >= 1 &&ret[0].Score > 0.05)
        for (unsigned int i = 1; i < ret.size(); i++)
        {
            //if (ret[i].Score > ret[0].Score * 0.3)
            if (ret[i].Score > 0.015)
            {          
                find_loop = true;
                int tmp_index = ret[i].Id;
                if (DEBUG_IMAGE && 0)
                {
                    auto it = image_pool.find(tmp_index);
                    cv::Mat tmp_image = (it->second).clone();
                    putText(tmp_image, "loop score:" + to_string(ret[i].Score), cv::Point2f(10, 50), CV_FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255));
                    cv::hconcat(loop_result, tmp_image, loop_result);
                }
            }

        }
/*
    if (DEBUG_IMAGE)
    {
        cv::imshow("loop_result", loop_result);
        cv::waitKey(20);
    }
*/
// 4.对于索引值大于50的关键帧才考虑进行回环,即系统开始的前50帧不进行回环
    //返回评分大于0.015的最早的关键帧索引min_index
    if (find_loop && frame_index > 50)
    {
        int min_index = -1;
        for (unsigned int i = 0; i < ret.size(); i++)
        {
            if (min_index == -1 || (ret[i].Id < min_index && ret[i].Score > 0.015))
                min_index = ret[i].Id;
        }
        return min_index;
    }
    else
        return -1;
}


void PoseGraph::addKeyFrameIntoVoc(KeyFrame* keyframe)
{
    // put image into image_pool; for visualization
    cv::Mat compressed_image;
    if (DEBUG_IMAGE)
    {
        int feature_num = keyframe->keypoints.size();
        cv::resize(keyframe->image, compressed_image, cv::Size(376, 240));
        putText(compressed_image, "feature_num:" + to_string(feature_num), cv::Point2f(10, 10), CV_FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255));
        image_pool[keyframe->index] = compressed_image;  //
    }

    db.add(keyframe->brief_descriptors);  //
}

// optimize4DoF 位姿图优化
void PoseGraph::optimize4DoF()
{
    while(true)
    {
        int cur_index = -1;
        int first_looped_index = -1;
        m_optimize_buf.lock();
        // 从优化队列当中获取最新的一个关键帧的index
        while(!optimize_buf.empty())
        {
            cur_index = optimize_buf.front();
            // earliest_loop_index当中存放的是数据库中第一个和滑动窗口中关键帧形成闭环的关键帧的index
            first_looped_index = earliest_loop_index;
            optimize_buf.pop();
        }
        m_optimize_buf.unlock();
        //optimize_buf中取出来的cur_index都是闭环帧的index
        if (cur_index != -1)
        {
            printf("optimize pose graph \n");
            TicToc tmp_t;
            m_keyframelist.lock();
            KeyFrame* cur_kf = getKeyFrame(cur_index);
            //max_length为要优化的变量最大个数
            int max_length = cur_index + 1;

            // w^t_i   w^q_i
            double t_array[max_length][3]; //平移数组，其中存放每个关键帧的平移向量
            Quaterniond q_array[max_length]; //旋转数组，其中存放每个关键帧的旋转四元数
            double euler_array[max_length][3];
            double sequence_array[max_length];
            // 构造四自由度位姿图优化
            ceres::Problem problem;
            ceres::Solver::Options options;
            options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
            //options.minimizer_progress_to_stdout = true;
            //options.max_solver_time_in_seconds = SOLVER_TIME * 3;
            options.max_num_iterations = 5;
            ceres::Solver::Summary summary;
            ceres::LossFunction *loss_function;
            loss_function = new ceres::HuberLoss(0.1);
            //loss_function = new ceres::CauchyLoss(1.0);
            //AngleLocalParameterization类的主要作用是指定yaw角优化变量的迭代更新，重载了括号运算
            ceres::LocalParameterization* angle_local_parameterization =
                AngleLocalParameterization::Create();

            list<KeyFrame*>::iterator it;

            int i = 0;
            //遍历关键帧列表
            for (it = keyframelist.begin(); it != keyframelist.end(); it++)
            {
                //first_looped_index为第一次闭环帧的index,需要优化的关键帧为从第一次闭环帧到当前帧间的所有关键帧
                if ((*it)->index < first_looped_index)
                    continue;
                (*it)->local_index = i;
                Quaterniond tmp_q;
                Matrix3d tmp_r;
                Vector3d tmp_t;
                //获取关键帧it的位姿
                (*it)->getVioPose(tmp_t, tmp_r);
                tmp_q = tmp_r;
                t_array[i][0] = tmp_t(0);
                t_array[i][1] = tmp_t(1);
                t_array[i][2] = tmp_t(2);
                q_array[i] = tmp_q;
                //将矩阵转换为向量
                Vector3d euler_angle = Utility::R2ypr(tmp_q.toRotationMatrix());
                euler_array[i][0] = euler_angle.x();
                euler_array[i][1] = euler_angle.y();
                euler_array[i][2] = euler_angle.z();

                sequence_array[i] = (*it)->sequence;
                //将关键帧列表中所有index>=first_looped_index的关键帧的位姿加入到参数块当中
                problem.AddParameterBlock(euler_array[i], 1, angle_local_parameterization);
                problem.AddParameterBlock(t_array[i], 3);
                //设置约束：如果该帧是最早的闭环帧的情况下，则固定它的位姿
                if ((*it)->index == first_looped_index || (*it)->sequence == 0)
                {   
                    problem.SetParameterBlockConstant(euler_array[i]);
                    problem.SetParameterBlockConstant(t_array[i]);
                }

                //add edge
                //add edge 这里添加的是序列边，是指通过VIO计算的两帧之间的相对位姿，每帧分别与其前边最多四帧构成序列边
                /**
                 * 顺序边的测量方程：p̂_{ij}^{i} = {R̂_i^w}^{-1} (p̂_j^w - p̂_i^w)
                 *                \hat{ψ}_ij = \hat{ψ}_j − \hat{ψ̂}_i
                 *  两个关键帧之间的相对位姿，由两个关键帧之间的VIO位姿估计变换得到
                 *   |------------------------------------|
                 *   |       |----------------------------|
                 *   |       |        |-------------------|
                 *   |       |        |         |---------|
                 * |帧1|    |帧2|    |帧3|     |帧4|     |帧5|
                */
                for (int j = 1; j < 5; j++)
                {
                  if (i - j >= 0 && sequence_array[i] == sequence_array[i-j])
                  {
                    Vector3d euler_conncected = Utility::R2ypr(q_array[i-j].toRotationMatrix());
                      //p̂_j^w - p̂_i^w 计算平移量的偏差
                    Vector3d relative_t(t_array[i][0] - t_array[i-j][0], t_array[i][1] - t_array[i-j][1], t_array[i][2] - t_array[i-j][2]);
                      //p̂_{ij}^{i} = {R̂_i^w}^{-1} (p̂_j^w - p̂_i^w)
                      //p̂^ij= (R̂^iw ) (p̂^jw − p̂^iw)
                    relative_t = q_array[i-j].inverse() * relative_t;
                      //ψ̂^_ij = ψ̂^_j − ψ̂^_i
                    double relative_yaw = euler_array[i][0] - euler_array[i-j][0];
                    ceres::CostFunction* cost_function = FourDOFError::Create( relative_t.x(), relative_t.y(), relative_t.z(),
                                                   relative_yaw, euler_conncected.y(), euler_conncected.z());
                    problem.AddResidualBlock(cost_function, NULL, euler_array[i-j], 
                                            t_array[i-j], 
                                            euler_array[i], 
                                            t_array[i]);
                  }
                }

                //add loop edge
                //add loop edge
                //这里添加的是闭环边，是指检测到闭环的两帧
                if((*it)->has_loop)
                {
                    assert((*it)->loop_index >= first_looped_index);
                    int connected_index = getKeyFrame((*it)->loop_index)->local_index;
                    Vector3d euler_conncected = Utility::R2ypr(q_array[connected_index].toRotationMatrix());
                    Vector3d relative_t;
                    relative_t = (*it)->getLoopRelativeT();
                    double relative_yaw = (*it)->getLoopRelativeYaw();
                    ceres::CostFunction* cost_function = FourDOFWeightError::Create( relative_t.x(), relative_t.y(), relative_t.z(),
                                                                               relative_yaw, euler_conncected.y(), euler_conncected.z());
                    problem.AddResidualBlock(cost_function, loss_function, euler_array[connected_index], 
                                                                  t_array[connected_index], 
                                                                  euler_array[i], 
                                                                  t_array[i]);
                    
                }
                
                if ((*it)->index == cur_index)
                    break;
                i++;
            }
            m_keyframelist.unlock();
            //开始优化
            ceres::Solve(options, &problem, &summary);
            //std::cout << summary.BriefReport() << "\n";
            
            //printf("pose optimization time: %f \n", tmp_t.toc());
            /*
            for (int j = 0 ; j < i; j++)
            {
                printf("optimize i: %d p: %f, %f, %f\n", j, t_array[j][0], t_array[j][1], t_array[j][2] );
            }
            */
            m_keyframelist.lock();
            //优化完成，使用优化后的位姿来更新关键帧列表中index大于等于first_looped_index的所有关键帧的位姿
            i = 0;
            for (it = keyframelist.begin(); it != keyframelist.end(); it++)
            {
                if ((*it)->index < first_looped_index)
                    continue;
                Quaterniond tmp_q;
                //向量转换为矩阵
                tmp_q = Utility::ypr2R(Vector3d(euler_array[i][0], euler_array[i][1], euler_array[i][2]));
                Vector3d tmp_t = Vector3d(t_array[i][0], t_array[i][1], t_array[i][2]);
                Matrix3d tmp_r = tmp_q.toRotationMatrix();
                (*it)-> updatePose(tmp_t, tmp_r);

                if ((*it)->index == cur_index)
                    break;
                i++;
            }

            Vector3d cur_t, vio_t;
            Matrix3d cur_r, vio_r;
            //获取优化后当前帧的位姿cur_t,cur_r
            cur_kf->getPose(cur_t, cur_r);
            //获取优化前有漂移的当前帧的位姿vio_t,vio_r
            cur_kf->getVioPose(vio_t, vio_r);
            //计算出当前帧的drift，更新全部关键帧位姿
            m_drift.lock();
            yaw_drift = Utility::R2ypr(cur_r).x() - Utility::R2ypr(vio_r).x();
            r_drift = Utility::ypr2R(Vector3d(yaw_drift, 0, 0));
            t_drift = cur_t - r_drift * vio_t;
            m_drift.unlock();
            //cout << "t_drift " << t_drift.transpose() << endl;
            //cout << "r_drift " << Utility::R2ypr(r_drift).transpose() << endl;
            //cout << "yaw drift " << yaw_drift << endl;

            it++;
            //下面代码为把当前关键帧it之后的关键帧的位姿通过求解的偏移量转换到world坐标系下
            for (; it != keyframelist.end(); it++)
            {
                Vector3d P;
                Matrix3d R;
                (*it)->getVioPose(P, R);
                P = r_drift * P + t_drift;
                R = r_drift * R;
                (*it)->updatePose(P, R);
            }
            m_keyframelist.unlock();
            //优化完后更新path
            updatePath();
        }

        std::chrono::milliseconds dura(2000);
        std::this_thread::sleep_for(dura);
    }
}

void PoseGraph::updatePath()
{
    m_keyframelist.lock();
    list<KeyFrame*>::iterator it;
    for (int i = 1; i <= sequence_cnt; i++)
    {
        path[i].poses.clear();
    }
    base_path.poses.clear();
    posegraph_visualization->reset();

    if (SAVE_LOOP_PATH)
    {
        ofstream loop_path_file_tmp(VINS_RESULT_PATH, ios::out);
        loop_path_file_tmp.close();
    }

    for (it = keyframelist.begin(); it != keyframelist.end(); it++)
    {
        Vector3d P;
        Matrix3d R;
        (*it)->getPose(P, R);
        Quaterniond Q;
        Q = R;
//        printf("path p: %f, %f, %f\n",  P.x(),  P.z(),  P.y() );

        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header.stamp = ros::Time((*it)->time_stamp);
        pose_stamped.header.frame_id = "world";
        pose_stamped.pose.position.x = P.x() + VISUALIZATION_SHIFT_X;
        pose_stamped.pose.position.y = P.y() + VISUALIZATION_SHIFT_Y;
        pose_stamped.pose.position.z = P.z();
        pose_stamped.pose.orientation.x = Q.x();
        pose_stamped.pose.orientation.y = Q.y();
        pose_stamped.pose.orientation.z = Q.z();
        pose_stamped.pose.orientation.w = Q.w();
        if((*it)->sequence == 0)
        {
            base_path.poses.push_back(pose_stamped);
            base_path.header = pose_stamped.header;
        }
        else
        {
            path[(*it)->sequence].poses.push_back(pose_stamped);
            path[(*it)->sequence].header = pose_stamped.header;
        }

        if (SAVE_LOOP_PATH)
        {
            ofstream loop_path_file(VINS_RESULT_PATH, ios::app);
            loop_path_file.setf(ios::fixed, ios::floatfield);
            loop_path_file.precision(0);
            loop_path_file << (*it)->time_stamp * 1e9 << ",";
            loop_path_file.precision(5);
            loop_path_file  << P.x() << ","
                  << P.y() << ","
                  << P.z() << ","
                  << Q.w() << ","
                  << Q.x() << ","
                  << Q.y() << ","
                  << Q.z() << ","
                  << endl;
            loop_path_file.close();
        }
        //draw local connection
        if (SHOW_S_EDGE)
        {
            list<KeyFrame*>::reverse_iterator rit = keyframelist.rbegin();
            list<KeyFrame*>::reverse_iterator lrit;
            for (; rit != keyframelist.rend(); rit++)  
            {  
                if ((*rit)->index == (*it)->index)
                {
                    lrit = rit;
                    lrit++;
                    for (int i = 0; i < 4; i++)
                    {
                        if (lrit == keyframelist.rend())
                            break;
                        if((*lrit)->sequence == (*it)->sequence)
                        {
                            Vector3d conncected_P;
                            Matrix3d connected_R;
                            (*lrit)->getPose(conncected_P, connected_R);
                            posegraph_visualization->add_edge(P, conncected_P);
                        }
                        lrit++;
                    }
                    break;
                }
            } 
        }
        if (SHOW_L_EDGE)
        {
            if ((*it)->has_loop && (*it)->sequence == sequence_cnt)
            {
                
                KeyFrame* connected_KF = getKeyFrame((*it)->loop_index);
                Vector3d connected_P;
                Matrix3d connected_R;
                connected_KF->getPose(connected_P, connected_R);
                //(*it)->getVioPose(P, R);
                (*it)->getPose(P, R);
                if((*it)->sequence > 0)
                {
                    posegraph_visualization->add_loopedge(P, connected_P + Vector3d(VISUALIZATION_SHIFT_X, VISUALIZATION_SHIFT_Y, 0));
                }
            }
        }

    }
    publish();
    m_keyframelist.unlock();
}


void PoseGraph::savePoseGraph()
{
    m_keyframelist.lock();
    TicToc tmp_t;
    FILE *pFile;
    printf("pose graph path: %s\n",POSE_GRAPH_SAVE_PATH.c_str());
    printf("pose graph saving... \n");
    string file_path = POSE_GRAPH_SAVE_PATH + "pose_graph.txt";
    pFile = fopen (file_path.c_str(),"w");
    //fprintf(pFile, "index time_stamp Tx Ty Tz Qw Qx Qy Qz loop_index loop_info\n");
    list<KeyFrame*>::iterator it;
    for (it = keyframelist.begin(); it != keyframelist.end(); it++)
    {
        std::string image_path, descriptor_path, brief_path, keypoints_path;
        if (DEBUG_IMAGE)
        {
            image_path = POSE_GRAPH_SAVE_PATH + to_string((*it)->index) + "_image.png";
            imwrite(image_path.c_str(), (*it)->image);
        }
        Quaterniond VIO_tmp_Q{(*it)->vio_R_w_i};
        Quaterniond PG_tmp_Q{(*it)->R_w_i};
        Vector3d VIO_tmp_T = (*it)->vio_T_w_i;
        Vector3d PG_tmp_T = (*it)->T_w_i;

        fprintf (pFile, " %d %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %d %f %f %f %f %f %f %f %f %d\n",(*it)->index, (*it)->time_stamp, 
                                    VIO_tmp_T.x(), VIO_tmp_T.y(), VIO_tmp_T.z(), 
                                    PG_tmp_T.x(), PG_tmp_T.y(), PG_tmp_T.z(), 
                                    VIO_tmp_Q.w(), VIO_tmp_Q.x(), VIO_tmp_Q.y(), VIO_tmp_Q.z(), 
                                    PG_tmp_Q.w(), PG_tmp_Q.x(), PG_tmp_Q.y(), PG_tmp_Q.z(), 
                                    (*it)->loop_index, 
                                    (*it)->loop_info(0), (*it)->loop_info(1), (*it)->loop_info(2), (*it)->loop_info(3),
                                    (*it)->loop_info(4), (*it)->loop_info(5), (*it)->loop_info(6), (*it)->loop_info(7),
                                    (int)(*it)->keypoints.size());

        // write keypoints, brief_descriptors   vector<cv::KeyPoint> keypoints vector<BRIEF::bitset> brief_descriptors;
        assert((*it)->keypoints.size() == (*it)->brief_descriptors.size());
        brief_path = POSE_GRAPH_SAVE_PATH + to_string((*it)->index) + "_briefdes.dat";
        std::ofstream brief_file(brief_path, std::ios::binary);
        keypoints_path = POSE_GRAPH_SAVE_PATH + to_string((*it)->index) + "_keypoints.txt";
        FILE *keypoints_file;
        keypoints_file = fopen(keypoints_path.c_str(), "w");
        for (int i = 0; i < (int)(*it)->keypoints.size(); i++)
        {
            brief_file << (*it)->brief_descriptors[i] << endl;
            fprintf(keypoints_file, "%f %f %f %f\n", (*it)->keypoints[i].pt.x, (*it)->keypoints[i].pt.y, 
                                                     (*it)->keypoints_norm[i].pt.x, (*it)->keypoints_norm[i].pt.y);
        }
        brief_file.close();
        fclose(keypoints_file);
    }
    fclose(pFile);

    printf("save pose graph time: %f s\n", tmp_t.toc() / 1000);
    m_keyframelist.unlock();
}


void PoseGraph::loadPoseGraph()
{
    TicToc tmp_t;
    FILE * pFile;
    string file_path = POSE_GRAPH_SAVE_PATH + "pose_graph.txt";
    printf("lode pose graph from: %s \n", file_path.c_str());
    printf("pose graph loading...\n");
    pFile = fopen (file_path.c_str(),"r");
    if (pFile == NULL)
    {
        printf("lode previous pose graph error: wrong previous pose graph path or no previous pose graph \n the system will start with new pose graph \n");
        return;
    }
    int index;
    double time_stamp;
    double VIO_Tx, VIO_Ty, VIO_Tz;
    double PG_Tx, PG_Ty, PG_Tz;
    double VIO_Qw, VIO_Qx, VIO_Qy, VIO_Qz;
    double PG_Qw, PG_Qx, PG_Qy, PG_Qz;
    double loop_info_0, loop_info_1, loop_info_2, loop_info_3;
    double loop_info_4, loop_info_5, loop_info_6, loop_info_7;
    int loop_index;
    int keypoints_num;
    Eigen::Matrix<double, 8, 1 > loop_info;
    int cnt = 0;
    while (fscanf(pFile,"%d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %d %lf %lf %lf %lf %lf %lf %lf %lf %d", &index, &time_stamp, 
                                    &VIO_Tx, &VIO_Ty, &VIO_Tz, 
                                    &PG_Tx, &PG_Ty, &PG_Tz, 
                                    &VIO_Qw, &VIO_Qx, &VIO_Qy, &VIO_Qz, 
                                    &PG_Qw, &PG_Qx, &PG_Qy, &PG_Qz, 
                                    &loop_index,
                                    &loop_info_0, &loop_info_1, &loop_info_2, &loop_info_3, 
                                    &loop_info_4, &loop_info_5, &loop_info_6, &loop_info_7,
                                    &keypoints_num) != EOF) 
    {
        /*
        printf("I read: %d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %d %lf %lf %lf %lf %lf %lf %lf %lf %d\n", index, time_stamp, 
                                    VIO_Tx, VIO_Ty, VIO_Tz, 
                                    PG_Tx, PG_Ty, PG_Tz, 
                                    VIO_Qw, VIO_Qx, VIO_Qy, VIO_Qz, 
                                    PG_Qw, PG_Qx, PG_Qy, PG_Qz, 
                                    loop_index,
                                    loop_info_0, loop_info_1, loop_info_2, loop_info_3, 
                                    loop_info_4, loop_info_5, loop_info_6, loop_info_7,
                                    keypoints_num);
        */
        cv::Mat image;
        std::string image_path, descriptor_path;
        if (DEBUG_IMAGE)
        {
            image_path = POSE_GRAPH_SAVE_PATH + to_string(index) + "_image.png";
            image = cv::imread(image_path.c_str(), 0);
        }

        Vector3d VIO_T(VIO_Tx, VIO_Ty, VIO_Tz);
        Vector3d PG_T(PG_Tx, PG_Ty, PG_Tz);
        Quaterniond VIO_Q;
        VIO_Q.w() = VIO_Qw;
        VIO_Q.x() = VIO_Qx;
        VIO_Q.y() = VIO_Qy;
        VIO_Q.z() = VIO_Qz;
        Quaterniond PG_Q;
        PG_Q.w() = PG_Qw;
        PG_Q.x() = PG_Qx;
        PG_Q.y() = PG_Qy;
        PG_Q.z() = PG_Qz;
        Matrix3d VIO_R, PG_R;
        VIO_R = VIO_Q.toRotationMatrix();
        PG_R = PG_Q.toRotationMatrix();
        Eigen::Matrix<double, 8, 1 > loop_info;
        loop_info << loop_info_0, loop_info_1, loop_info_2, loop_info_3, loop_info_4, loop_info_5, loop_info_6, loop_info_7;

        if (loop_index != -1)
            if (earliest_loop_index > loop_index || earliest_loop_index == -1)
            {
                earliest_loop_index = loop_index;
            }

        // load keypoints, brief_descriptors   
        string brief_path = POSE_GRAPH_SAVE_PATH + to_string(index) + "_briefdes.dat";
        std::ifstream brief_file(brief_path, std::ios::binary);
        string keypoints_path = POSE_GRAPH_SAVE_PATH + to_string(index) + "_keypoints.txt";
        FILE *keypoints_file;
        keypoints_file = fopen(keypoints_path.c_str(), "r");
        vector<cv::KeyPoint> keypoints;
        vector<cv::KeyPoint> keypoints_norm;
        vector<BRIEF::bitset> brief_descriptors;
        for (int i = 0; i < keypoints_num; i++)
        {
            BRIEF::bitset tmp_des;
            brief_file >> tmp_des;
            brief_descriptors.push_back(tmp_des);
            cv::KeyPoint tmp_keypoint;
            cv::KeyPoint tmp_keypoint_norm;
            double p_x, p_y, p_x_norm, p_y_norm;
            if(!fscanf(keypoints_file,"%lf %lf %lf %lf", &p_x, &p_y, &p_x_norm, &p_y_norm))
                printf(" fail to load pose graph \n");
            tmp_keypoint.pt.x = p_x;
            tmp_keypoint.pt.y = p_y;
            tmp_keypoint_norm.pt.x = p_x_norm;
            tmp_keypoint_norm.pt.y = p_y_norm;
            keypoints.push_back(tmp_keypoint);
            keypoints_norm.push_back(tmp_keypoint_norm);
        }
        brief_file.close();
        fclose(keypoints_file);

        KeyFrame* keyframe = new KeyFrame(time_stamp, index, VIO_T, VIO_R, PG_T, PG_R, image, loop_index, loop_info, keypoints, keypoints_norm, brief_descriptors);
        loadKeyFrame(keyframe, 0);
        if (cnt % 20 == 0)
        {
            publish();
        }
        cnt++;
    }
    fclose (pFile);
    printf("load pose graph time: %f s\n", tmp_t.toc()/1000);
    base_sequence = 0;
}

void PoseGraph::publish()
{
    for (int i = 1; i <= sequence_cnt; i++)
    {
        //if (sequence_loop[i] == true || i == base_sequence)
        if (1 || i == base_sequence)
        {
            pub_pg_path.publish(path[i]);
            pub_path[i].publish(path[i]);
            posegraph_visualization->publish_by(pub_pose_graph, path[sequence_cnt].header);
        }
    }
    base_path.header.frame_id = "world";
    pub_base_path.publish(base_path);
    //posegraph_visualization->publish_by(pub_pose_graph, path[sequence_cnt].header);
}

void PoseGraph::updateKeyFrameLoop(int index, Eigen::Matrix<double, 8, 1 > &_loop_info)
{
    KeyFrame* kf = getKeyFrame(index);
    kf->updateLoop(_loop_info);
    if (abs(_loop_info(7)) < 30.0 && Vector3d(_loop_info(0), _loop_info(1), _loop_info(2)).norm() < 20.0)
    {
        if (FAST_RELOCALIZATION)
        {
            KeyFrame* old_kf = getKeyFrame(kf->loop_index);
            Vector3d w_P_old, w_P_cur, vio_P_cur;
            Matrix3d w_R_old, w_R_cur, vio_R_cur;
            old_kf->getPose(w_P_old, w_R_old);
            kf->getVioPose(vio_P_cur, vio_R_cur);

            Vector3d relative_t;
            Quaterniond relative_q;
            relative_t = kf->getLoopRelativeT();
            relative_q = (kf->getLoopRelativeQ()).toRotationMatrix();
            w_P_cur = w_R_old * relative_t + w_P_old;
            w_R_cur = w_R_old * relative_q;
            double shift_yaw;
            Matrix3d shift_r;
            Vector3d shift_t; 
            shift_yaw = Utility::R2ypr(w_R_cur).x() - Utility::R2ypr(vio_R_cur).x();
            shift_r = Utility::ypr2R(Vector3d(shift_yaw, 0, 0));
            shift_t = w_P_cur - w_R_cur * vio_R_cur.transpose() * vio_P_cur; 

            m_drift.lock();
            yaw_drift = shift_yaw;
            r_drift = shift_r;
            t_drift = shift_t;
            m_drift.unlock();
        }
    }
}