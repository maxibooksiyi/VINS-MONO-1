#include "initial_sfm.h"

GlobalSFM::GlobalSFM(){}

/** @brief 三角化两帧间某个对应特征点的深度
 * @param[in]  Pose和point
 * @param[out] point_3d
 */
// 线性三角化法DLT triangulatePoint()已知两帧对应的2D点(归一化坐标)和两帧的相机位姿(以左图为参考)，求解路标3D坐标(三维场景)
void GlobalSFM::triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
						Vector2d &point0, Vector2d &point1, Vector3d &point_3d)
{
	Matrix4d design_matrix = Matrix4d::Zero();
	// 三角化原理得出的公式
	design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
	design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
	design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
	design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
	Vector4d triangulated_point;
	// svd分解
	triangulated_point =
		      design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
	point_3d(0) = triangulated_point(0) / triangulated_point(3);
	point_3d(1) = triangulated_point(1) / triangulated_point(3);
	point_3d(2) = triangulated_point(2) / triangulated_point(3);
}

/** @brief PNP方法得到第l帧到第i帧的位姿
 * @param[in]  i  第i帧
 * @param[update]  R_initial、P_initial、sfm_f
 */
bool GlobalSFM::solveFrameByPnP(Matrix3d &R_initial, Vector3d &P_initial, int i,
								vector<SFMFeature> &sfm_f)
{
	vector<cv::Point2f> pts_2_vector;
	vector<cv::Point3f> pts_3_vector;
	for (int j = 0; j < feature_num; j++)  //遍历所有特征点
	{
		if (sfm_f[j].state != true)  // 检查状态，该点是否被三角化
			continue;
		Vector2d point2d;
		for (int k = 0; k < (int)sfm_f[j].observation.size(); k++)
		{
			if (sfm_f[j].observation[k].first == i)
			{
				Vector2d img_pts = sfm_f[j].observation[k].second;
				cv::Point2f pts_2(img_pts(0), img_pts(1));
				pts_2_vector.push_back(pts_2);
				cv::Point3f pts_3(sfm_f[j].position[0], sfm_f[j].position[1], sfm_f[j].position[2]);
				pts_3_vector.push_back(pts_3);
				break;
			}
		}
	}
	if (int(pts_2_vector.size()) < 15)
	{
		printf("unstable features tracking, please slowly move you device!\n");
		if (int(pts_2_vector.size()) < 10)
			return false;
	}
	cv::Mat r, rvec, t, D, tmp_r;
	cv::eigen2cv(R_initial, tmp_r);  //转换成solvePnP能处理的格式
	/** void Rodrigues( const CvMat* src，CvMat* dst，CvMat* jacobian=0 );
        src——为输入的旋转向量（3x1或者1x3）或者旋转矩阵（3x3） 该参数向量表示其旋转的角度，用向量长度表示
        dst——为输出的旋转矩阵（3x3）或者旋转向量（3x1或者1x3）
        jacobian——为可选的输出雅可比矩阵（3x9或者9x3） 是输入与输出数组的偏导数
	 */
	cv::Rodrigues(tmp_r, rvec);   //罗德里格斯公式
	cv::eigen2cv(P_initial, t);
	// 内参数矩阵
	cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
	bool pnp_succ;
    //得到了第l帧到第i帧的旋转平移
    /** void solvePnP(InputArray objectPoints, InputArray imagePoints, InputArray cameraMatrix, InputArray distCoeffs, OutputArray rvec, OutputArray tvec, bool useExtrinsicGuess=false, int flags = CV_ITERATIVE)
        objectPoints：世界坐标系中的3D点坐标，单位mm
        imagePoints：图像坐标系中点的坐标，单位像素
        cameraMatrix：相机内参矩阵
        distCoeffs：畸变系数
        rvec：旋转矩阵
        tvec：平移矩阵
        useExtrinsicGuess：是否输出平移矩阵和旋转矩阵，默认为false
        flags：SOLVEPNP _ITERATIVE、SOLVEPNP _P3P、SOLVEPNP _EPNP、SOLVEPNP _DLS、- SOLVEPNP _UPNP
     */
	pnp_succ = cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1); ////rvec t 作为参考的初始位姿且提供初始位姿
	if(!pnp_succ)
	{
		return false;
	}
	cv::Rodrigues(rvec, r);
	MatrixXd R_pnp;
	cv::cv2eigen(r, R_pnp);  //转换成原有格式
	MatrixXd T_pnp;
	cv::cv2eigen(t, T_pnp);
	R_initial = R_pnp;
	P_initial = T_pnp;
	return true;

}
// triangulateTwoFrames（）函数三角化两帧特征点，更新sfm_f.position
/** @brief 三角化frame0和frame1间所有对应点
 * @param[in]  frame,Pose 帧索引和位姿数据
 * @param[out]   sfm_f的state和position 3D坐标
 */
void GlobalSFM::triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0, 
									 int frame1, Eigen::Matrix<double, 3, 4> &Pose1,
									 vector<SFMFeature> &sfm_f)
{
	assert(frame0 != frame1);
	for (int j = 0; j < feature_num; j++)  //
	{
		if (sfm_f[j].state == true)  // 没有三角化的话才开始三角化
			continue;
		bool has_0 = false, has_1 = false;
		Vector2d point0;
		Vector2d point1;
		for (int k = 0; k < (int)sfm_f[j].observation.size(); k++)  // 找到对应的帧索引,取出2D像素坐标
		{
			if (sfm_f[j].observation[k].first == frame0)
			{
				point0 = sfm_f[j].observation[k].second;
				has_0 = true;
			}
			if (sfm_f[j].observation[k].first == frame1)
			{
				point1 = sfm_f[j].observation[k].second;
				has_1 = true;
			}
		}
		if (has_0 && has_1) // 如果2D像素坐标都匹配上了,三角化
		{
			Vector3d point_3d;
			triangulatePoint(Pose0, Pose1, point0, point1, point_3d);
			sfm_f[j].state = true;
			sfm_f[j].position[0] = point_3d(0);
			sfm_f[j].position[1] = point_3d(1);
			sfm_f[j].position[2] = point_3d(2);
			//cout << "trangulated : " << frame1 << "  3d point : "  << j << "  " << point_3d.transpose() << endl;
		}							  
	}
}


/**为了求解滑动窗内所有图像帧相对于第l帧（参考帧）的位姿和三角化特征点3D坐标
 * 输出所有图像帧相对于参考帧l的姿态四元数Q 平移向量T 特征点坐标,放在sfm_f和sfm_tracked_points内*/
/**
 * @brief   纯视觉sfm，求解窗口中的所有图像帧相对于第l帧的位姿和三角化的特征点坐标
 * @param[in]   frame_num	窗口总帧数（frame_count + 1）
 * @param[out]  q 	窗口内图像帧的旋转四元数q（相对于第l帧）Q[frame_count + 1]
 * @param[out]	T 	窗口内图像帧的平移向量T（相对于第l帧）
 * @param[in]  	l 	第l帧
 * @param[in]  	relative_R	第l帧到当前帧的旋转矩阵
 * @param[in]  	relative_T 	第l帧到当前帧的平移向量
 * @param[in]  	sfm_f		所有特征点
 * @param[out]  sfm_tracked_points 所有在sfm中三角化的特征点ID和坐标
 * @return  bool true:sfm求解成功
*/


// relative_q[i][j]  j_q_i
// relative_t[i][j]  j_t_ji  (j < i)
bool GlobalSFM::construct(int frame_num, Quaterniond* q, Vector3d* T, int l,
			  const Matrix3d relative_R, const Vector3d relative_T,
			  vector<SFMFeature> &sfm_f, map<int, Vector3d> &sfm_tracked_points)
{
	feature_num = sfm_f.size();  // 特征点数目
	// cout << "set 0 and " << l << " as known " << endl;
	// have relative_r relative_t
	// intial two view
    // 假设第l帧为原点,根据当前帧到第l帧的relative_R,relative_T,得到当前帧位姿
    // 	q w_R_cam T w_t_cam
    q[l].w() = 1;
	q[l].x() = 0;
	q[l].y() = 0;
	q[l].z() = 0;
	T[l].setZero();
	q[frame_num - 1] = q[l] * Quaterniond(relative_R);  //当前帧
	T[frame_num - 1] = relative_T;

	//rotate to cam frame
	Matrix3d c_Rotation[frame_num];
	Vector3d c_Translation[frame_num];
	Quaterniond c_Quat[frame_num];
	//  c_rotation cam_R_w  c_translation cam_t_w
	double c_rotation[frame_num][4];
	double c_translation[frame_num][3];

	Eigen::Matrix<double, 3, 4> Pose[frame_num];  // 位姿3*4，第l帧到每一帧变换矩阵
    // 初始化了第l帧和第frame_num-1帧的姿态,以第l帧的姿态作为世界坐标系
    // 第l帧
	c_Quat[l] = q[l].inverse();
	c_Rotation[l] = c_Quat[l].toRotationMatrix();
	c_Translation[l] = -1 * (c_Rotation[l] * T[l]);
	Pose[l].block<3, 3>(0, 0) = c_Rotation[l];
	Pose[l].block<3, 1>(0, 3) = c_Translation[l];
    // 滑动窗最后一帧，即当前帧
	c_Quat[frame_num - 1] = q[frame_num - 1].inverse();
	c_Rotation[frame_num - 1] = c_Quat[frame_num - 1].toRotationMatrix();
	c_Translation[frame_num - 1] = -1 * (c_Rotation[frame_num - 1] * T[frame_num - 1]);
	Pose[frame_num - 1].block<3, 3>(0, 0) = c_Rotation[frame_num - 1];
	Pose[frame_num - 1].block<3, 1>(0, 3) = c_Translation[frame_num - 1];


	//1: trangulate between l ----- frame_num - 1
	//2: solve pnp l + 1; trangulate l + 1 ------- frame_num - 1;
	// 1) pnp得到 l,l+1,l+2...frmaenum-2相机位姿，三角化l,l+1,l+2...frmaenum-2和framenum-1帧
	for (int i = l; i < frame_num - 1 ; i++)
	{
		// solve pnp
		if (i > l)  // l+1
		{
			Matrix3d R_initial = c_Rotation[i - 1];
			Vector3d P_initial = c_Translation[i - 1];
            // 求解PnP得到 l+1,l+2...frmaenum-2相机位姿R T
			if(!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
				return false;
			c_Rotation[i] = R_initial;
			c_Translation[i] = P_initial;
			c_Quat[i] = c_Rotation[i];
			Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
			Pose[i].block<3, 1>(0, 3) = c_Translation[i];
		}
		// triangulate point based on the solve pnp result
        // l,l+1,l+2.....framenum-2和framenum-1 三角化恢复路标3D坐标(三维场景)
		triangulateTwoFrames(i, Pose[i], frame_num - 1, Pose[frame_num - 1], sfm_f);
	}
	//3: triangulate l-----l+1 l+2 ... frame_num -2
    // 2) 三角化 l+1,l+2......framenum-2和l帧
	for (int i = l + 1; i < frame_num - 1; i++)
		triangulateTwoFrames(l, Pose[l], i, Pose[i], sfm_f);
	//4: solve pnp l-1; triangulate l-1 ----- l
	//             l-2              l-2 ----- l
	// 3) pnp得到l-1,l-2.....0相机位姿,三角化 l-1, l-2, l-3, …, 0帧与l帧
	for (int i = l - 1; i >= 0; i--)
	{
		//solve pnp
		Matrix3d R_initial = c_Rotation[i + 1];
		Vector3d P_initial = c_Translation[i + 1];
		if(!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
			return false;
		c_Rotation[i] = R_initial;
		c_Translation[i] = P_initial;
		c_Quat[i] = c_Rotation[i];
		Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
		Pose[i].block<3, 1>(0, 3) = c_Translation[i];
		//triangulate
		triangulateTwoFrames(i, Pose[i], l, Pose[l], sfm_f);
	}
	//5: triangulate all other points
	// 4) 剩下的没有三角化的特征点,通过它被观察的第一帧和最后一帧进行三角定位
	for (int j = 0; j < feature_num; j++)
	{
		if (sfm_f[j].state == true)
			continue;
		if ((int)sfm_f[j].observation.size() >= 2)
		{
			Vector2d point0, point1;
			int frame_0 = sfm_f[j].observation[0].first;
			point0 = sfm_f[j].observation[0].second;
			int frame_1 = sfm_f[j].observation.back().first;
			point1 = sfm_f[j].observation.back().second;
			Vector3d point_3d;
			triangulatePoint(Pose[frame_0], Pose[frame_1], point0, point1, point_3d);
			sfm_f[j].state = true;
			sfm_f[j].position[0] = point_3d(0);
			sfm_f[j].position[1] = point_3d(1);
			sfm_f[j].position[2] = point_3d(2);
			//cout << "trangulated : " << frame_0 << " " << frame_1 << "  3d point : "  << j << "  " << point_3d.transpose() << endl;
		}		
	}

/*
	for (int i = 0; i < frame_num; i++)
	{
		q[i] = c_Rotation[i].transpose(); 
		cout << "solvePnP  q" << " i " << i <<"  " <<q[i].w() << "  " << q[i].vec().transpose() << endl;
	}
	for (int i = 0; i < frame_num; i++)
	{
		Vector3d t_tmp;
		t_tmp = -1 * (q[i] * c_Translation[i]);
		cout << "solvePnP  t" << " i " << i <<"  " << t_tmp.x() <<"  "<< t_tmp.y() <<"  "<< t_tmp.z() << endl;
	}
*/
	//full BA
	// 5) 使用ceres进行全局BA优化（相机位姿和特征点坐标）
	// 由于使用的优化参数为四元数，需要使用LocalParameterization将四元数重构为三维旋转矢量进行优化和更新，就必须使用AddParameterBlock()显式传递参数
	ceres::Problem problem;
	ceres::LocalParameterization* local_parameterization = new ceres::QuaternionParameterization();
	// begin full BA
	// 加入待优化量:全局位姿
	for (int i = 0; i < frame_num; i++)
	{
		//double array for ceres
		// 优化位姿,没有优化特征点3D坐标
		c_translation[i][0] = c_Translation[i].x();
		c_translation[i][1] = c_Translation[i].y();
		c_translation[i][2] = c_Translation[i].z();
		c_rotation[i][0] = c_Quat[i].w();
		c_rotation[i][1] = c_Quat[i].x();
		c_rotation[i][2] = c_Quat[i].y();
		c_rotation[i][3] = c_Quat[i].z();
		problem.AddParameterBlock(c_rotation[i], 4, local_parameterization);  //四元数重构为三维旋转矢量进行优化和更新
		problem.AddParameterBlock(c_translation[i], 3);
		// c.固定先验值  因为l帧是参考系,最新帧的平移也是先验,如果不固定住,原本可观的量会变的不可观
		if (i == l)
		{
			problem.SetParameterBlockConstant(c_rotation[i]);
		}
		if (i == l || i == frame_num - 1)
		{
			problem.SetParameterBlockConstant(c_translation[i]);
		}
	}
    // d.加入残差块 这里采用的是最小化重投影误差的方式(Bundle Adjustment),所以需要2D-3D信息,注意这块没有加loss function
	for (int i = 0; i < feature_num; i++)
	{
		if (sfm_f[i].state != true)
			continue;
		for (int j = 0; j < int(sfm_f[i].observation.size()); j++)
		{
			int l = sfm_f[i].observation[j].first;
			ceres::CostFunction* cost_function = ReprojectionError3D::Create(
												sfm_f[i].observation[j].second.x(),   // 像素坐标u
												sfm_f[i].observation[j].second.y());  // 像素坐标v
    		problem.AddResidualBlock(cost_function, NULL, c_rotation[l], c_translation[l], 
    								sfm_f[i].position);	 
		}
	}
	// e.schur消元求解
	// schur消元有2大作用:一个是在最小二乘中利用H矩阵稀疏的性质进行加速求解，另一个是在sliding window时求解marg掉老帧后的先验信息矩阵
    ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR;   //schur分解，用于BA问题求解
	//options.minimizer_progress_to_stdout = true;
	options.max_solver_time_in_seconds = 0.2; //求解器的最大运行秒数
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	//std::cout << summary.BriefReport() << "\n";
	if (summary.termination_type == ceres::CONVERGENCE || summary.final_cost < 5e-03)
	{
		//cout << "vision only BA converge" << endl;
	}
	else
	{
		//cout << "vision only BA not converge " << endl;
		return false;
	}
	// d．返回特征点l系下3D坐标和优化后的全局位姿
	for (int i = 0; i < frame_num; i++)
	{
		q[i].w() = c_rotation[i][0]; 
		q[i].x() = c_rotation[i][1]; 
		q[i].y() = c_rotation[i][2]; 
		q[i].z() = c_rotation[i][3]; 
		q[i] = q[i].inverse();   //各帧到l帧的旋转平移
		//cout << "final  q" << " i " << i <<"  " <<q[i].w() << "  " << q[i].vec().transpose() << endl;
	}
	for (int i = 0; i < frame_num; i++)
	{
		T[i] = -1 * (q[i] * Vector3d(c_translation[i][0], c_translation[i][1], c_translation[i][2]));
		//cout << "final  t" << " i " << i <<"  " << T[i](0) <<"  "<< T[i](1) <<"  "<< T[i](2) << endl;
	}
	for (int i = 0; i < (int)sfm_f.size(); i++)
	{
		if(sfm_f[i].state)
			sfm_tracked_points[sfm_f[i].id] = Vector3d(sfm_f[i].position[0], sfm_f[i].position[1], sfm_f[i].position[2]);
	}
	return true;
}

