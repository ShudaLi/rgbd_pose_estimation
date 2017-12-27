#ifndef BTL_P3P_POSE_HEADER
#define BTL_P3P_POSE_HEADER

#include <Eigen/Dense>
#include "PnPPoseAdapter.hpp"

using namespace Eigen;
using namespace cv;

template< typename POSE_T >
std::vector<POSE_T> o4_roots(const Matrix<POSE_T, Dynamic, Dynamic> & p_)
{
	POSE_T A = p_(0, 0);
	POSE_T B = p_(1, 0);
	POSE_T C = p_(2, 0);
	POSE_T D = p_(3, 0);
	POSE_T E = p_(4, 0);

	POSE_T A_pw2 = A*A;
	POSE_T B_pw2 = B*B;
	POSE_T A_pw3 = A_pw2*A;
	POSE_T B_pw3 = B_pw2*B;
	POSE_T A_pw4 = A_pw3*A;
	POSE_T B_pw4 = B_pw3*B;

	POSE_T alpha = -3 * B_pw2 / (8 * A_pw2) + C / A;
	POSE_T beta = B_pw3 / (8 * A_pw3) - B*C / (2 * A_pw2) + D / A;
	POSE_T gamma = -3 * B_pw4 / (256 * A_pw4) + B_pw2*C / (16 * A_pw3) - B*D / (4 * A_pw2) + E / A;

	POSE_T alpha_pw2 = alpha*alpha;
	POSE_T alpha_pw3 = alpha_pw2*alpha;

	std::complex<POSE_T> P(-alpha_pw2 / 12 - gamma, 0);
	std::complex<POSE_T> Q(-alpha_pw3 / 108 + alpha*gamma / 3 - pow(beta, 2) / 8, 0);
	std::complex<POSE_T> R = -Q / POSE_T(2.0) + sqrt(pow(Q, POSE_T(2.)) / POSE_T(4.) + pow(P, POSE_T(3.)) / POSE_T(27.));

	std::complex<POSE_T> U = pow(R, POSE_T(1.0 / 3.0));
	std::complex<POSE_T> y;

	if (U.real() == 0)
		y = -POSE_T(5.0)*alpha / POSE_T(6.) - pow(Q, POSE_T(1.0 / 3.0));
	else
		y = -POSE_T(5.0)*alpha / POSE_T(6.) - P / (POSE_T(3.)*U) + U;

	std::complex<POSE_T> w = sqrt(alpha + POSE_T(2.)*y);

	std::vector<POSE_T> realRoots;
	std::complex<POSE_T> temp;
	temp = -B / (POSE_T(4.)*A) + POSE_T(0.5)*(w + sqrt(-(POSE_T(3.)*alpha + POSE_T(2.)*y + POSE_T(2.)*beta / w)));
	realRoots.push_back(temp.real());
	temp = -B / (POSE_T(4.)*A) + POSE_T(0.5)*(w - sqrt(-(POSE_T(3.)*alpha + POSE_T(2.)*y + POSE_T(2.)*beta / w)));
	realRoots.push_back(temp.real());
	temp = -B / (POSE_T(4.)*A) + POSE_T(0.5)*(-w + sqrt(-(POSE_T(3.)*alpha + POSE_T(2.)*y - POSE_T(2.)*beta / w)));
	realRoots.push_back(temp.real());
	temp = -B / (POSE_T(4.)*A) + POSE_T(0.5)*(-w - sqrt(-(POSE_T(3.)*alpha + POSE_T(2.)*y - POSE_T(2.)*beta / w)));
	realRoots.push_back(temp.real());

	return realRoots;
}


template< typename POSE_T, typename POINT_T > 
void kneip_main(const Matrix<POINT_T, Dynamic, Dynamic>& X_w, const Matrix<POINT_T, Dynamic, Dynamic>& bv, vector< Sophus::SE3<POSE_T> >* p_solutions_){

	p_solutions_->clear();

	Matrix<POSE_T, 3, 1> P1 = X_w.col(0).template cast<POSE_T>();
	Matrix<POSE_T, 3, 1> P2 = X_w.col(1).template cast<POSE_T>();
	Matrix<POSE_T, 3, 1> P3 = X_w.col(2).template cast<POSE_T>();

	Matrix<POSE_T, 3, 1> temp1 = P2 - P1;
	Matrix<POSE_T, 3, 1> temp2 = P3 - P1;

	if (temp1.cross(temp2).norm() == 0)
		return;

	Matrix<POSE_T, 3, 1> f1 = bv.col(0).template cast<POSE_T>();
	Matrix<POSE_T, 3, 1> f2 = bv.col(1).template cast<POSE_T>();
	Matrix<POSE_T, 3, 1> f3 = bv.col(2).template cast<POSE_T>();

	Matrix<POSE_T, 3, 1> e1 = f1;
	Matrix<POSE_T, 3, 1> e3 = f1.cross(f2);
	e3 = e3 / e3.norm();
	Matrix<POSE_T, 3, 1> e2 = e3.cross(e1);

	Matrix<POSE_T, 3, 3> RR;
	RR.row(0) = e1.transpose();
	RR.row(1) = e2.transpose();
	RR.row(2) = e3.transpose();

	f3 = RR*f3;

	if (f3(2, 0) > 0)
	{
		f1 = bv.col(1).template cast<POSE_T>();
		f2 = bv.col(0).template cast<POSE_T>();
		f3 = bv.col(2).template cast<POSE_T>();

		e1 = f1;
		e3 = f1.cross(f2);
		e3 = e3 / e3.norm();
		e2 = e3.cross(e1);

		RR.row(0) = e1.transpose();
		RR.row(1) = e2.transpose();
		RR.row(2) = e3.transpose();

		f3 = RR*f3;

		P1 = X_w.col(1).template cast<POSE_T>();
		P2 = X_w.col(0).template cast<POSE_T>();
		P3 = X_w.col(2).template cast<POSE_T>();
	}

	Matrix<POSE_T, 3, 1> n1 = P2 - P1;
	n1 = n1 / n1.norm();
	Matrix<POSE_T, 3, 1> n3 = n1.cross(P3 - P1);
	n3 = n3 / n3.norm();
	Matrix<POSE_T, 3, 1> n2 = n3.cross(n1);

	Matrix<POSE_T, 3, 3> N;
	N.row(0) = n1.transpose();
	N.row(1) = n2.transpose();
	N.row(2) = n3.transpose();

	P3 = N*(P3 - P1);

	POSE_T d_12 = temp1.norm();
	POSE_T f_1 = f3(0, 0) / f3(2, 0);
	POSE_T f_2 = f3(1, 0) / f3(2, 0);
	POSE_T p_1 = P3(0, 0);
	POSE_T p_2 = P3(1, 0);

	POSE_T cos_beta = f1.dot(f2);
	POSE_T b = 1 / (1 - pow(cos_beta, 2)) - 1;

	if (cos_beta < 0)
		b = -sqrt(b);
	else
		b = sqrt(b);

	POSE_T f_1_pw2 = pow(f_1, 2);
	POSE_T f_2_pw2 = pow(f_2, 2);
	POSE_T p_1_pw2 = pow(p_1, 2);
	POSE_T p_1_pw3 = p_1_pw2 * p_1;
	POSE_T p_1_pw4 = p_1_pw3 * p_1;
	POSE_T p_2_pw2 = pow(p_2, 2);
	POSE_T p_2_pw3 = p_2_pw2 * p_2;
	POSE_T p_2_pw4 = p_2_pw3 * p_2;
	POSE_T d_12_pw2 = pow(d_12, 2);
	POSE_T b_pw2 = pow(b, 2);

	Eigen::Matrix<POSE_T, 5, 1> factors;

	factors(0, 0) = -f_2_pw2*p_2_pw4
		- p_2_pw4*f_1_pw2
		- p_2_pw4;

	factors(1, 0) = 2 * p_2_pw3*d_12*b
		+ 2 * f_2_pw2*p_2_pw3*d_12*b
		- 2 * f_2*p_2_pw3*f_1*d_12;

	factors(2, 0) = -f_2_pw2*p_2_pw2*p_1_pw2
		- f_2_pw2*p_2_pw2*d_12_pw2*b_pw2
		- f_2_pw2*p_2_pw2*d_12_pw2
		+ f_2_pw2*p_2_pw4
		+ p_2_pw4*f_1_pw2
		+ 2 * p_1*p_2_pw2*d_12
		+ 2 * f_1*f_2*p_1*p_2_pw2*d_12*b
		- p_2_pw2*p_1_pw2*f_1_pw2
		+ 2 * p_1*p_2_pw2*f_2_pw2*d_12
		- p_2_pw2*d_12_pw2*b_pw2
		- 2 * p_1_pw2*p_2_pw2;

	factors(3, 0) = 2 * p_1_pw2*p_2*d_12*b
		+ 2 * f_2*p_2_pw3*f_1*d_12
		- 2 * f_2_pw2*p_2_pw3*d_12*b
		- 2 * p_1*p_2*d_12_pw2*b;

	factors(4, 0) = -2 * f_2*p_2_pw2*f_1*p_1*d_12*b
		+ f_2_pw2*p_2_pw2*d_12_pw2
		+ 2 * p_1_pw3*d_12
		- p_1_pw2*d_12_pw2
		+ f_2_pw2*p_2_pw2*p_1_pw2
		- p_1_pw4
		- 2 * f_2_pw2*p_2_pw2*p_1*d_12
		+ p_2_pw2*f_1_pw2*p_1_pw2
		+ f_2_pw2*p_2_pw2*d_12_pw2*b_pw2;

	std::vector<POSE_T> realRoots = o4_roots<POSE_T>(factors);

	for (int i = 0; i < 4; i++)
	{
		POSE_T cot_alpha =
			(-f_1*p_1 / f_2 - realRoots[i] * p_2 + d_12*b) /
			(-f_1*realRoots[i] * p_2 / f_2 + p_1 - d_12);

		POSE_T cos_theta = realRoots[i];
		POSE_T sin_theta = sqrt(1 - pow(realRoots[i], 2));
		POSE_T sin_alpha = sqrt(1 / (pow(cot_alpha, 2) + 1));
		POSE_T cos_alpha = sqrt(1 - pow(sin_alpha, 2));

		if (cot_alpha < 0)
			cos_alpha = -cos_alpha;

		Matrix<POSE_T, 3, 1> C;
		C(0, 0) = d_12*cos_alpha*(sin_alpha*b + cos_alpha);
		C(1, 0) = cos_theta*d_12*sin_alpha*(sin_alpha*b + cos_alpha);
		C(2, 0) = sin_theta*d_12*sin_alpha*(sin_alpha*b + cos_alpha);

		C = P1 + N.transpose() *C;

		Matrix<POSE_T, 3, 3> R;
		R(0, 0) = -cos_alpha;
		R(0, 1) = -sin_alpha*cos_theta;
		R(0, 2) = -sin_alpha*sin_theta;
		R(1, 0) = sin_alpha;
		R(1, 1) = -cos_alpha*cos_theta;
		R(1, 2) = -cos_alpha*sin_theta;
		R(2, 0) = 0.0;
		R(2, 1) = -sin_theta;
		R(2, 2) = cos_theta;

		//R = N.transpose()*R.transpose()*RR;
		R = RR.transpose()*R*N;

		p_solutions_->push_back(Sophus::SE3<POSE_T>(Sophus::SO3<POSE_T>(R), Sophus::SE3<POSE_T>::Point(-R*C)));
	}
}

template< typename POSE_T, typename POINT_T > /*Matrix<float,-1,-1,0,-1,-1> = MatrixXf*/
vector< Sophus::SE3<POSE_T> > kneip(PnPPoseAdapter<POSE_T, POINT_T>& adapter, int i0 = 0, int i1 = 1, int i2 = 2){
	Matrix<POINT_T, Dynamic, Dynamic> bv(3,3);
	bv.col(0) = adapter.getBearingVector(i0);
	bv.col(1) = adapter.getBearingVector(i1);
	bv.col(2) = adapter.getBearingVector(i2);
	Matrix<POINT_T, Dynamic, Dynamic> X_w(3, 3);
	X_w.col(0) = adapter.getPointGlob(i0);
	X_w.col(1) = adapter.getPointGlob(i1);
	X_w.col(2) = adapter.getPointGlob(i2);
	
	vector< Sophus::SE3<POSE_T> > solutions;
	kneip_main<POSE_T, POINT_T>(X_w, bv, &solutions);
	return solutions;
}

template< typename POSE_T, typename POINT_T > 
bool kneip(const Matrix<POINT_T, Dynamic, Dynamic>& X_w_, const Matrix<POINT_T, Dynamic, Dynamic>& bv_, 
	Sophus::SE3<POSE_T>* p_sol_){
	typedef Matrix<POINT_T, 3, 1> Point3;

	vector< Sophus::SE3<POSE_T> > v_solutions;
	kneip_main<POSE_T, POINT_T>(X_w_, bv_, &v_solutions);
	
	POINT_T minScore = numeric_limits<POINT_T>::max();
	int minIndex = -1;
	for (int i = 0; i < v_solutions.size(); i++)
	{
		Point3 pc = v_solutions[i].so3().template cast<POINT_T>() * X_w_.col(3) + v_solutions[i].translation().template cast<POINT_T>();// transform pw into pc

		//compute the score
		//Vector3 E = pc - adapter.getPointCurr(selected_cols[0]);
		//T dist = E.norm();
		//check for best solution
		//if (dist < minScore) {
		//	minScore = dist;
		//	minIndex = i;
		//}

		pc = pc / pc.norm(); //normalize pc

		//compute the score
		POINT_T score = 1.0 - pc.transpose() * bv_.col(3);

		//check for best solution
		if (score < minScore) {
			minScore = score;
			minIndex = i;
		}
	}


	if (minIndex != -1){
		*p_sol_ = v_solutions[minIndex];
		return true;
	}
	else
	{
		return false;
	}
}

template<typename T>
int RANSACUpdateNumIters(T p, T ep, const int modelPoints, const int maxIters)
{
	p = std::max(p, T(0.));
	p = std::min(p, T(1.));
	ep = std::max(ep, T(0.));
	ep = std::min(ep, T(1.));

	// avoid inf's & nan's
	T num = std::max(T(1. - p), std::numeric_limits<T>::epsilon());
	T denom = T(1.) - std::pow(T(1. - ep), modelPoints);
	if (denom < std::numeric_limits<T>::epsilon())
		return 0;

	num = std::log(num);
	denom = std::log(denom);

	return denom >= 0 || -num >= maxIters*(-denom) ? maxIters : int(num / denom+0.5f);
}

template< typename POSE_T, typename POINT_T > /*Eigen::Matrix<float,-1,-1,0,-1,-1> = Eigen::MatrixXf*/
void kneip_ransac( PnPPoseAdapter<POSE_T, POINT_T>& adapter,	const POINT_T thre_2d_, int& Iter, POINT_T confidence = 0.99){

	POSE_T cos_thr = cos(atan(thre_2d_ / adapter.getFocal()));

	typedef Sophus::SE3<POSE_T> RT;
	typedef Matrix<POINT_T, 3, 1> Point3;

	RandomElements<int> re(adapter.getNumberCorrespondences());
	const int K = 4;
	adapter.setMaxVotes(-1);
	for (int i = 0; i < Iter; i++)	{
		//randomly select K candidates
		vector<int> selected_cols;
		re.run(K, &selected_cols);

		vector< RT > solutions = kneip<POSE_T, POINT_T>(adapter, selected_cols[0], selected_cols[1], selected_cols[2]);
		//use the fourth point to verify the estimation.
		POINT_T minScore = 1000000.0;
		int minIndex = -1;
		for (int i = 0; i < solutions.size(); i++)
		{
			Point3 pw = adapter.getPointGlob(selected_cols[3]);
			Point3 pc = solutions[i].so3().template cast<POINT_T>() * pw + solutions[i].translation().template cast<POINT_T>();// transform pw into pc
			pc = pc / pc.norm(); //normalize pc

			//compute the score
			POINT_T score = 1.0 - pc.transpose() * adapter.getBearingVector(selected_cols[3]);

			//check for best solution
			if (score < minScore) {
				minScore = score;
				minIndex = i;
			}
		}

		if (minIndex != -1){
			const RT& outModel = solutions[minIndex];

			Matrix<short, Dynamic, Dynamic> inliers(adapter.getNumberCorrespondences(), 1);
			inliers.setZero();
			int votes = 0;
			for (int i = 0; i < adapter.getNumberCorrespondences(); i++)
			{
				Point3 Xw = adapter.getPointGlob(i);
				Point3 Xc = outModel.so3().template cast<POINT_T>() * Xw + outModel.translation().template cast<POINT_T>();// transform pw into pc
				Xc = Xc / Xc.norm(); //normalize pc

				//compute the score
				POINT_T cos_a = Xc.dot( adapter.getBearingVector(i) );

				//check for best solution
				if (cos_a > cos_thr) {
					inliers(i) = 1;
					votes++;
				}
			}

			if (votes > adapter.getMaxVotes()){
				adapter.setMaxVotes(votes);
				adapter.setRcw(outModel.so3());
				adapter.sett(outModel.translation());
				adapter.setInlier(inliers);
				//Iter = RANSACUpdateNumIters(confidence, (POINT_T)(adapter.getNumberCorrespondences() - votes) / adapter.getNumberCorrespondences(), K+1, Iter);
			}
		}
		//update iterations
	}
	adapter.cvtInlier();

	return;
}

template< typename POSE_T, typename POINT_T > /*Eigen::Matrix<float,-1,-1,0,-1,-1> = Eigen::MatrixXf*/
void lsq_pnp( PnPPoseAdapter<POSE_T, POINT_T>& adapter,	int& Iter){
	typedef Sophus::SE3<POSE_T> RT;
	typedef Matrix<POINT_T, 3, 1> Point3;

	int total = adapter.getNumberCorrespondences();
	POSE_T total_err = 0.;
	//POSE_T* pErr = new POSE_T[total_err];
	for (int i = 0; i < total; i++)
	{
		Point3 Xc = adapter.getRcw()*adapter.getPointGlob(i) + adapter.gettw();
		Xc.normalize();
		POINT_T err = Xc.cross( adapter.getBearingVector(i) ).norm();
		total_err += err;
	}

	//int itr = 0;
	//while (itr>=2){
	//	itr = ceil(itr / 2);
	//	for (int i = 0; i < itr; i++)
	//	{
	//		if (itr * 2 + i < total){
	//			pErr[i] += pErr[itr * 2 + i];

	//		}
	//	}
	//}
	cout << total_err << endl;

	return;
}


#endif
