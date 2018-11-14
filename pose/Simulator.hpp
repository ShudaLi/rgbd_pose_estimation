#ifndef _RAND_GENERATOR_HEADER_
#define _RAND_GENERATOR_HEADER_

#include <se3.hpp>
#include <time.h>
#include <limits>
#include <random>
#include "Utility.hpp"

using namespace Eigen;
using namespace std;

std::default_random_engine generator;
std::normal_distribution<double> distribution(0., 1.0);

template< typename T >
Matrix<T, 3, 1> generate_random_translation_uniform(T size)
{
	Matrix<T, 3, 1>  translation = Matrix< T, Dynamic, Dynamic>::Random(3, 1);
	return size * translation;
}

template< typename T >
Sophus::SO3< T > generate_random_rotation(T max_angle_radian_, bool use_guassian_ = true)
{
	Matrix<T, 3, 1> rv;
	if (use_guassian_){
		rv[0] = distribution(generator); //standard normal distribution
		rv[1] = distribution(generator);
		rv[2] = distribution(generator);
	}
	else {
		rv = Matrix< T, Dynamic, Dynamic>::Random(3, 1); //uniform -1. to  1.
	}

	rv[0] = max_angle_radian_*rv[0]; //angle rotation around x
	rv[1] = max_angle_radian_*rv[1] * T(.5); // y axis
	rv[2] = max_angle_radian_*rv[2]; // z axis

	T m_pi_2 = T(M_PI / 2.);
	rv[0] = rv[0] > M_PI ? M_PI : rv[0];
	rv[0] = rv[0] < -M_PI ? -M_PI : rv[0];
	rv[1] = rv[1] > m_pi_2 ? m_pi_2 : rv[1];
	rv[1] = rv[1] < -m_pi_2 ? -m_pi_2 : rv[1];
	rv[2] = rv[2] > M_PI ? M_PI : rv[2];
	rv[2] = rv[2] < -M_PI ? -M_PI : rv[2];

	Matrix<T, 3, 3>  R1;
	R1(0, 0) = 1.0;
	R1(0, 1) = 0.0;
	R1(0, 2) = 0.0;
	R1(1, 0) = 0.0;
	R1(1, 1) = cos(rv[0]);
	R1(1, 2) = -sin(rv[0]);
	R1(2, 0) = 0.0;
	R1(2, 1) = -R1(1, 2); //sin(rpy[0]);
	R1(2, 2) = R1(1, 1); //cos(rpy[0])

	Matrix<T, 3, 3>  R2;
	R2(0, 0) = cos(rv[1]);
	R2(0, 1) = 0.0;
	R2(0, 2) = sin(rv[1]);
	R2(1, 0) = 0.0;
	R2(1, 1) = 1.0;
	R2(1, 2) = 0.0;
	R2(2, 0) = -R2(0, 2);
	R2(2, 1) = 0.0;
	R2(2, 2) = R2(0, 0);

	Matrix<T, 3, 3>  R3;
	R3(0, 0) = cos(rv[2]);
	R3(0, 1) = -sin(rv[2]);
	R3(0, 2) = 0.0;
	R3(1, 0) = -R3(0, 1);
	R3(1, 1) = R3(0, 0);
	R3(1, 2) = 0.0;
	R3(2, 0) = 0.0;
	R3(2, 1) = 0.0;
	R3(2, 2) = 1.0;

	Sophus::SO3< T > rotation(R3 * R2 * R1);
	return rotation;
}

template< typename T >
void simulate_nl_nl_correspondences(const Sophus::SO3<T>& R_cw_, int number_, T noise_nl_, T outlier_ratio_nl_, bool use_guassian_,
	Matrix<T, Dynamic, Dynamic>* pM_, Matrix<T, Dynamic, Dynamic>* pN_, Matrix<T, Dynamic, Dynamic>* pN_gt = NULL,
	Matrix<T, Dynamic, Dynamic>* p_all_weights_ = NULL)
{
	typedef Matrix<T, 3, 1> Point3;
	typedef Matrix<T, Dynamic, Dynamic> MX;

	MX w(number_, 1); //dynamic weights for N-N correspondences
	pM_->resize(3, number_); //normal in WRS
	pN_->resize(3, number_); //normal in CRS
	MX N_gt(3, number_); //ground truth normal in CRS
	for (int i = 0; i < number_; i++)	{
		do {
			N_gt.col(i) = generate_random_rotation<T>( T(M_PI / 2.), false) * Point3(0, 0, -1); //have to be uniform distribution here
			N_gt.col(i).normalize();
			pM_->col(i) = R_cw_.inverse() * N_gt.col(i); //transform to world reference system
			pM_->col(i).normalize(); 
			//add noise
			pN_->col(i) = generate_random_rotation<T>(noise_nl_, use_guassian_) * N_gt.col(i);
			pN_->col(i).normalize();
		} while (acos(pN_->col(i)(2)) < T(M_PI / 2)); //ensure that the normal is facing the camera
		//simulate weights
		w(i) = pN_->col(i).dot(N_gt.col(i));
	}
	//add outliers
	int out = int(outlier_ratio_nl_*number_ + T(.5));
	RandomElements<int> re(number_);
	vector<int> vIdx;	re.run(out, &vIdx);
	for (int i = 0; i < out; i++){
		Matrix< T, 3, 1> nl_g;
		do {
			pN_->col(i) = generate_random_rotation<T>(M_PI / 2, false) * Point3(0, 0, -1);//have to be uniform distribution here
			pN_->col(i).normalize();
		} while (acos(pN_->col(i)(2)) < M_PI / 2);
	}

	if (pN_gt){
		*pN_gt = N_gt; //note that p_nl_c_gt_ is not polluted by noise or outliers
	}
	if (p_all_weights_){
		assert(p_all_weights_->rows() == number_ && p_all_weights_->cols() == 3);
		p_all_weights_->col(2) = w;
	}
	return;
}

//generate random 3-D point within a viewing frutum defined by
//T min_depth_, T max_depth_, 
//T tan_fov_x, T tan_fov_y
template< typename T >
Matrix<T, 3, 1> generate_a_random_point(T min_depth_, T max_depth_, T tan_fov_x, T tan_fov_y)
{
	T x_range = tan_fov_x * max_depth_;
	T y_range = tan_fov_y * max_depth_;
	Matrix<T, 3, 1>  cleanPoint = Matrix<T, 3, 1>::Random(3, 1);
	cleanPoint[0] *= x_range; //x
	cleanPoint[1] *= y_range; //y
	cleanPoint[2] = (cleanPoint[2] + 1.) / 2. *(max_depth_ - min_depth_) + min_depth_; //z
	return cleanPoint;
}

//project a set of 3-D points in camera coordinate system to 2-D points
template< typename T >
Matrix<T, Dynamic, Dynamic> project_point_cloud(const Matrix<T, Dynamic, Dynamic>& pt_c, T f_){
	Matrix<T, Dynamic, Dynamic> pt_2d(2, pt_c.cols());
	for (int i = 0; i < pt_c.cols(); i++) {
		pt_2d.col(i)[0] = f_ * pt_c.col(i)[0] / pt_c.col(i)[2];
		pt_2d.col(i)[1] = f_ * pt_c.col(i)[1] / pt_c.col(i)[2];
	}
	return pt_2d;
}

template< typename T >
Matrix<T, Dynamic, Dynamic> simulate_rand_point_cloud_in_frustum(int number_, T f_, T min_depth_, T max_depth_)
{
	T tan_fov_x = T(320. / f_); //the frame resolution is 640x480 with principle point at the centre of the frame
	T tan_fov_y = T(240. / f_);
	Matrix<T, Dynamic, Dynamic> all_P(3, number_);
	for (int i = 0; i < (int)number_; i++){
		bool is_outside_frustum = true;
		while (is_outside_frustum){
			Matrix<T, 3, 1 > P = generate_a_random_point<T>(min_depth_, max_depth_, tan_fov_x, tan_fov_y); // generate random point in CRS
			all_P.col(i) = P;
			is_outside_frustum = !(fabs(P[0] / P[2]) < tan_fov_x && fabs(P[1] / P[2]) < tan_fov_y); //check whether the point is inside the frustum
		}
	}
	return all_P;
}

template< typename T >
void simulate_2d_3d_correspondences(const Sophus::SO3<T>& R_cw_, const Matrix<T, 3, 1>& t_w_,
	int number_, T noise_, T outlier_ratio_, T min_depth_, T max_depth_, T f_, bool use_guassian_,
	Matrix<T, Dynamic, Dynamic>* pQ_, // *pQ_: points with noise in 3-D world system 
	Matrix<T, Dynamic, Dynamic>* pU_, // *pU_: 2-D points 
	Matrix<T, Dynamic, Dynamic>* pP_gt = NULL, // *pP_gt: ground truth of 3-D points in camera system 
	Matrix<T, Dynamic, Dynamic>* p_all_weights_ = NULL)
{
	typedef Matrix<T, Dynamic, Dynamic> MX;

	MX w(number_, 1); //dynamic weights for 2-3 correspondences
	//1. generate 3-D points P in CRS
	MX P_gt = simulate_rand_point_cloud_in_frustum<T>(number_, f_, min_depth_, max_depth_); //pt cloud in camera system
	//2. project to 2-D 
	MX kp_2d = project_point_cloud<T>(P_gt, f_);
	//3. transform from world to camera coordinate system
	pQ_->resize(3, number_);
	for (int i = 0; i < number_; i++) {
		pQ_->col(i) = R_cw_.inverse() * (P_gt.col(i) - t_w_);
	}
	//4. add 2-D noise
	for (int i = 0; i < number_; i++) {
		Matrix<T, 2, 1> rv; //random variable
		if (use_guassian_)
			rv = Matrix<T, 2, 1>(distribution(generator), distribution(generator));
		else
			rv = MX::Random(2, 1);
		w(i) = T(1.) / rv.norm();
		kp_2d.col(i) = kp_2d.col(i) + noise_ * rv;
	}

	//5. add 2-D outliers
	int out = int(outlier_ratio_*number_ + .5);
	MX out_points = simulate_rand_point_cloud_in_frustum<T>(out, f_, min_depth_, max_depth_); //outliers remain in CRS
	MX out_2d_points = project_point_cloud<T>(out_points, f_);
	RandomElements<int> re(number_);
	vector<int> vIdx;	re.run(out, &vIdx);
	for (int i = 0; i < out; i++){
		kp_2d.col(vIdx[i]) = out_2d_points.col(i);
	}
	//6. convert to 2-D key points into unit vectors 
	pU_->resize(3, number_); 
	pU_->row(0) = kp_2d.row(0);
	pU_->row(1) = kp_2d.row(1);
	pU_->row(2) = f_ * MX::Ones(1,number_);

	for (int c = 0; c < number_; c++) {
		pU_->col(c).normalize();
	}
	if (pP_gt){
		*pP_gt = P_gt; //note that pt_c was not polluted by outliers
	}

	if (p_all_weights_){
		assert(p_all_weights_->rows() == number_ && p_all_weights_->cols() == 3);
		p_all_weights_->col(0) = w;
	}
	return;
}


template< typename T >
void simulate_3d_3d_correspondences(const Sophus::SO3<T>& R_cw_, const Matrix<T, 3, 1>& t_w_,
	int number_, T noise_, T outlier_ratio_, T min_depth_, T max_depth_, T f_, bool use_guassian_,
	Matrix<T, Dynamic, Dynamic>* pQ_, // *pQ_: points with noise in 3-D world system 
	Matrix<T, Dynamic, Dynamic>* pP_gt = NULL, // *pP_gt: ground truth of 3-D points in camera system 
	Matrix<T, Dynamic, Dynamic>* p_all_weights_ = NULL)
{
	typedef Matrix<T, Dynamic, Dynamic> MX;

	MX w(number_, 1); //dynamic weights for 2-3 correspondences
	//1. generate 3-D points P in CRS
	MX P_gt = simulate_rand_point_cloud_in_frustum<T>(number_, f_, min_depth_, max_depth_); //pt cloud in camera system
	//3. transform from camera to world coordinate system
	pQ_->resize(3, number_);
	for (int i = 0; i < number_; i++) {
		pQ_->col(i) = R_cw_.inverse() * (P_gt.col(i) - t_w_);
	}
	//4. add 3-D noise
	for (int i = 0; i < number_; i++) {
		Matrix<T, 3, 1> rv; //random variable
		if (use_guassian_)
			rv = Matrix<T, 3, 1>(distribution(generator), distribution(generator), distribution(generator));
		else
			rv = MX::Random(3, 1);
		w(i) = T(1.) / rv.norm();
		pQ_->col(i) = pQ_->col(i) + noise_*rv;
	}

	//5. add 2-D outliers
	int out = int(outlier_ratio_*number_ + .5);
	MX out_points = simulate_rand_point_cloud_in_frustum<T>(out, f_, min_depth_, max_depth_); //outliers remain in CRS
	RandomElements<int> re(number_);
	vector<int> vIdx;	re.run(out, &vIdx);
	for (int i = 0; i < out; i++){
		pQ_->col(vIdx[i]) = out_points.col(i);
	}
	
	if (pP_gt){
		*pP_gt = P_gt; //note that pt_c was not polluted by outliers
	}

	if (p_all_weights_){
		assert(p_all_weights_->rows() == number_ && p_all_weights_->cols() == 3);
		p_all_weights_->col(1) = w;
	}
	return;
}

template< typename T >
void simulate_2d_3d_nl_correspondences(const Sophus::SO3<T>& R_cw_, const Matrix<T, 3, 1>& t_w_, //rotation and translation in world reference system
	int number_,  // total number of correspondences
	T n2D_, T or_2D_, //noise level and outlier ratio for 2-3 correspondences
	T n3D_, T or_3D_, //noise level and outlier ratio for 3-3 correspondences
	T nNl_, T or_Nl_, //noise level and outlier ratio for N-N correspondences
	T min_depth_, T max_depth_, T f_, bool use_guassian_,
	//outputs
	Matrix<T, Dynamic, Dynamic>* pQ_, Matrix<T, Dynamic, Dynamic>* pM_, //Q and M are 3-D points and normal in world 
	Matrix<T, Dynamic, Dynamic>* pP_, Matrix<T, Dynamic, Dynamic>* pN_, //P and N are 3-D points and normal in camera system
	Matrix<T, Dynamic, Dynamic>* pU_,//U are the unit vectors pointing from camera centre to 2-D key points
	Matrix<T, Dynamic, Dynamic>* p_all_weights_ = NULL) //store all weights for 2-3, 3-3 and N-N correspondences
{
	Matrix<T, Dynamic, Dynamic> all_weights(number_, 3);
	//generate 2-D to 3-D pairs
	Matrix<T, Dynamic, Dynamic> P_gt; //ground truth 3-D points in camera reference system
	simulate_2d_3d_correspondences<T>(R_cw_, t_w_, number_, n2D_, or_2D_, min_depth_, max_depth_, f_, use_guassian_,
		&*pQ_, &*pU_, &P_gt, &all_weights);

	//generate normal to normal pairs
	Matrix<T, Dynamic, Dynamic> nl_c_gt;
	simulate_nl_nl_correspondences<T>(R_cw_, number_, nNl_, or_Nl_, true, &*pM_, &*pN_, &nl_c_gt, &all_weights);

	//add noise to 3-D points in camera reference frame
	pP_->resize(3, number_);
	for (int i = 0; i < number_; i++) {
		Matrix<T, 3, 1> rv; //random variable
		if (use_guassian_) //use Guassian noise
			rv = Matrix<T, 3, 1>(distribution(generator), distribution(generator), distribution(generator));
		else
			rv = Matrix<T, Dynamic, Dynamic>::Random(3, 1);
		//all_weights(i, 1) = short(rv.norm() / 1.414 * numeric_limits<short>::max()); //simulate weights
		all_weights(i, 1) = T(1.) / rv.norm();
		pP_->col(i) = P_gt.col(i) + n3D_ * rv;
	}

	//add 3-D outliers
	int out = int(or_3D_*number_ + .5);
	RandomElements<int> re(number_);
	vector<int> vIdx;	re.run(out, &vIdx);
	Matrix<T, Dynamic, Dynamic> pt_c_out = simulate_rand_point_cloud_in_frustum(out, f_, min_depth_, max_depth_);
	for (int i = 0; i < out; i++){
		pP_->col(vIdx[i]) = pt_c_out.col(i);
	}

	if (p_all_weights_){
		assert(p_all_weights_->rows() == number_ && p_all_weights_->cols() == 3);
		*p_all_weights_ = all_weights;
	}

	return;
}

template< typename T >
T lateral_noise_kinect(T theta_, T z_, T f_)
{
	//Nguyen, C.V., Izadi, S., &Lovell, D. (2012).Modeling kinect sensor noise for improved 3D reconstruction and tracking.In 3DIM / 3DPVT (pp. 524?30).http://doi.org/10.1109/3DIMPVT.2012.84
	T sigma_l;
	sigma_l = T(.8) + T(.035)*theta_ / (T(M_PI / 2.) - theta_);
	sigma_l = sigma_l * z_ / f_;
	return sigma_l;
}

template< typename T >
T axial_noise_kinect(T theta_, T z_){
	T sigma_a;
	if (fabs(theta_) <= T(M_PI / 3.))
		sigma_a = T(.0012) + T(.0019)*(z_ - 0.4)*(z_ - 0.4);
	else
		sigma_a = T(.0012) + T(.0019)*(z_ - 0.4)*(z_ - 0.4) + T(.0001) * theta_* theta_ / sqrt(z_) / (M_PI / 2 - theta_) / (M_PI / 2 - theta_);
	return sigma_a;
}

template< typename T >
void simulate_kinect_2d_3d_nl_correspondences(const Sophus::SO3<T>& R_cw_, const Matrix<T, 3, 1>& t_w_, int number_, 
	T noise_2d_, T outlier_ratio_2d_, T outlier_ratio_3d_, T noise_nl_, T outlier_ratio_nl_, T min_depth_, T max_depth_, T f_,
	Matrix<T, Dynamic, Dynamic>* p_pt_w_, 
	Matrix<T, Dynamic, Dynamic>* p_nl_w_,
	Matrix<T, Dynamic, Dynamic>* p_pt_c_, 
	Matrix<T, Dynamic, Dynamic>* p_nl_c_, 
	Matrix<T, Dynamic, Dynamic>* p_bv_,
	Matrix<T, Dynamic, Dynamic>* p_weights_ = NULL)
{
	Matrix<T, Dynamic, Dynamic> all_weights(number_, 3); //simulated weights
	Matrix<T, Dynamic, Dynamic> pt_c_gt; //ground truth 3-D points in camera reference system
	simulate_2d_3d_correspondences<T>(R_cw_, t_w_, number_, noise_2d_, outlier_ratio_2d_, min_depth_, max_depth_, f_, true,
							&*p_pt_w_, &*p_bv_, &pt_c_gt, &all_weights);

	//generate normal to normal pairs
	Matrix<T, Dynamic, Dynamic> nl_c_gt;
	simulate_nl_nl_correspondences<T>(R_cw_, number_, noise_nl_, outlier_ratio_nl_, true, &*p_nl_w_, &*p_nl_c_, &nl_c_gt, &all_weights);

	T sigma_min = axial_noise_kinect<T>(T(.0), min_depth_);

	//add Gaussian noise to 3-D points in camera reference frame
	p_pt_c_->resize(3, number_);
	for (int i = 0; i < number_; i++) {
		T theta = acos(nl_c_gt.col(i).dot(Matrix<T, 3, 1>(0, 0, -1)));
		T z = pt_c_gt.col(i)(2);
		T sigma_l = lateral_noise_kinect<T>(theta, z, f_);
		T sigma_a = axial_noise_kinect<T>(theta, z);
		Matrix<T, 3, 1> random_variable(sigma_l*distribution(generator), sigma_l*distribution(generator), sigma_a*distribution(generator));
		p_pt_c_->col(i) = pt_c_gt.col(i) + random_variable;
		all_weights(i, 1) = T(sigma_min / sigma_a);
	}

	//add 3-D outliers
	int out = int(outlier_ratio_3d_*number_ + .5);
	RandomElements<int> re(number_);
	vector<int> vIdx;	re.run(out, &vIdx);
	Matrix<T, Dynamic, Dynamic> pt_c_out = simulate_rand_point_cloud_in_frustum(out, f_, min_depth_, max_depth_);
	for (int i = 0; i < out; i++){
		p_pt_c_->col(vIdx[i]) = pt_c_out.col(i);
	}
	if (p_weights_){
		assert(p_weights_->rows() == number_ && p_weights_->cols() == 3);
		*p_weights_ = all_weights;
	}

	return;
}


#endif
