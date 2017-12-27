#ifndef BTL_AO_POSE_HEADER
#define BTL_AO_POSE_HEADER

#include <Eigen/Dense>
#include "AOPoseAdapter.hpp"
#include "P3P.hpp"

using namespace Eigen;
using namespace cv;

template< typename Tp >
Matrix<Tp, 2, 1> calc_percentage_err(const Sophus::SO3<Tp>& R_cw_, const Matrix<Tp, 3, 1>& t_w_, const PoseAdapterBase<Tp>* p_ad){
	Matrix<Tp, 3, 1> te = R_cw_*t_w_ - p_ad->getRcw() * p_ad->gettw();
	Tp t_e = te.norm() / p_ad->gettw().norm() * 100;

	//Quaternion<POSE_T> q_gt(R_cw_.matrix());
	Quaternion<Tp> q_gt = R_cw_.unit_quaternion();
	Quaternion<Tp> q_est = p_ad->getRcw().unit_quaternion();

	q_gt.w() -= q_est.w();
	q_gt.x() -= q_est.x();
	q_gt.y() -= q_est.y();
	q_gt.z() -= q_est.z();

	Tp r_e = q_gt.norm() / q_est.norm() * 100;
	return Matrix<Tp, 2, 1>(t_e, r_e);
}

template< typename Tp >
Sophus::SE3<Tp> shinji(
	const Matrix<Tp, Dynamic, Dynamic> & X_w_, 
	const Matrix<Tp, Dynamic, Dynamic>&  X_c_, int K){
	// X_c_ = R_cw * X_w_ + Tw; //R_cw and tw is defined in world and transform a point in world to camera coordinate system
	// main references: Shinji, U. (1991). Least-squares estimation of transformation parameters between two point patterns. PAMI.
	assert(3 <= K && K <= X_w_.cols() && X_w_.cols() == X_c_.cols() && X_w_.rows() == X_c_.rows() && X_w_.rows() == 3);
	//Compute the centroid of each point set
	Matrix<Tp, 3, 1> Cw(0, 0, 0), Cc(0, 0, 0); //Matrix<float,3,1,0,3,1> = Vector3f
	for (int nCount = 0; nCount < K; nCount++){
		Cw += X_w_.col(nCount).template cast<Tp>();
		Cc += X_c_.col(nCount).template cast<Tp>();
	}
	Cw /= (Tp)K;
	Cc /= (Tp)K;

	// transform coordinate
	Matrix<Tp, 3, 1> Aw, Ac;
	Matrix<Tp, 3, 3> M; M.setZero();
	Matrix<Tp, 3, 3> N;
	Tp sigma_w_sqr = 0, sigma_c_sqr = 0;
	for (int nC = 0; nC < K; nC++){
		Aw = X_w_.col(nC).template cast<Tp>() - Cw; sigma_w_sqr += Aw.norm();
		Ac = X_c_.col(nC).template cast<Tp>() - Cc; sigma_c_sqr += Ac.norm();
		N = Ac * Aw.transpose();
		M += N;
	}

	M /= (Tp) X_w_.cols() ;
	sigma_w_sqr /= (Tp)K;
	sigma_c_sqr /= (Tp)K;

	JacobiSVD<Matrix<Tp, -1, -1, 0, -1, -1> > svd(M, ComputeFullU | ComputeFullV);
	//[U S V]=svd(M);
	//R=U*V';
	Matrix<Tp, 3, 3> U = svd.matrixU();
	Matrix<Tp, 3, 3> V = svd.matrixV();
	Matrix<Tp, 3, 1> D = svd.singularValues();
	Sophus::SO3<Tp> R_tmp;
	if (M.determinant() < 0){
		Matrix<Tp, 3, 3> I = Matrix<Tp, 3, 3>::Identity(); I(2, 2) = -1; D(2) *= -1;
		R_tmp = Sophus::SO3<Tp>(U*I*V.transpose());
	}
	else{
		R_tmp = Sophus::SO3<Tp>(U*V.transpose());
	}
	Matrix< Tp, 3, 1> T_tmp = Cc - R_tmp * Cw;
	Sophus::SE3<Tp> solution(R_tmp, T_tmp);
	
	return solution;
}

template< typename Tp >
void shinji_ransac(AOPoseAdapter<Tp>& adapter, 
	const Tp dist_thre_3d_, int& Iter, Tp confidence = 0.99){
	typedef Matrix<Tp, Dynamic, Dynamic> MatrixX;
	typedef Matrix<Tp, 3, 1> Point3;
	typedef Sophus::SE3<Tp> RT;

	RandomElements<int> re((int)adapter.getNumberCorrespondences());
	const int K = 3;

	adapter.setMaxVotes(-1);
	for (int ii = 0; ii < Iter; ii++)	{
		//randomly select K candidates
		vector<int> selected_cols;
		re.run(K, &selected_cols);
		MatrixX eimX_world(3, K), eimX_cam(3, K), bv(3, K);
		bool invalid_sample = false;
		for (int nSample = 0; nSample < K; nSample++) {
			eimX_world.col(nSample) = adapter.getPointGlob(selected_cols[nSample]);
			if (adapter.isValid(selected_cols[nSample])){
				eimX_cam.col(nSample) = adapter.getPointCurr(selected_cols[nSample]);
			}
			else{
				invalid_sample = true;
			}
		}

		if (invalid_sample) continue;
		//calc R&t		
		RT* p_solution; RT solution; 

		solution = shinji<Tp>(eimX_world, eimX_cam, K);
			
		//collect votes
		int votes = 0;
		Matrix<short, Dynamic, Dynamic> inliers(adapter.getNumberCorrespondences(),2); inliers.setZero();
		for (int c = 0; c < adapter.getNumberCorrespondences(); c++) {
			if (adapter.isValid(c)){
				Point3 eivE = adapter.getPointCurr(c) - (solution.so3().template cast<Tp>() * adapter.getPointGlob(c) + solution.translation().template cast<Tp>());
				if (eivE.norm() < dist_thre_3d_){
					inliers(c,1) = 1;
					votes++;
				}
			}
		}
		//cout << endl;

		if (votes > adapter.getMaxVotes()){
			adapter.setMaxVotes( votes );
			adapter.setRcw(solution.so3());
			adapter.sett(solution.translation());
			adapter.setInlier(inliers);
			//Iter = RANSACUpdateNumIters(confidence, (T)(adapter.getNumberCorrespondences() - votes) / adapter.getNumberCorrespondences(), K, Iter);
		}
	}
	adapter.cvtInlier();

	return;
}

template< typename Tp >
void shinji_ls(AOPoseAdapter<Tp>& adapter){
	typedef Matrix<Tp, Dynamic, Dynamic> MatrixX;
	typedef Matrix<Tp, 3, 1> Point3;
	typedef Sophus::SE3<Tp> RT;

	const vector<short>& vInliers = adapter.getInlierIdx();
	int K = vInliers.size();
	MatrixX Xw(3, K);
	MatrixX Xc(3, K);

	for (int ii = 0; ii < K; ii++)	{
		short idx = vInliers[ii];
		Xw.col(ii) = adapter.getPointGlob(idx);
		Xc.col(ii) = adapter.getPointCurr(idx);
	}

	RT solution = shinji<Tp>(Xw, Xc, K);

	adapter.setRcw(solution.so3());
	adapter.sett(solution.translation());

	return;
}

template< typename Tp >
bool assign_sample(const AOPoseAdapter<Tp>& adapter, const vector<int>& selected_cols_, 
	Matrix<Tp, Dynamic, Dynamic>* p_X_w_, Matrix<Tp, Dynamic, Dynamic>* p_X_c_, Matrix<Tp, Dynamic, Dynamic>* p_bv_){
	int K = (int)selected_cols_.size() - 1;
	bool use_shinji = false;
	int nValid = 0;
	for (int nSample = 0; nSample < K; nSample++) {
		p_X_w_->col(nSample) = adapter.getPointGlob(selected_cols_[nSample]);
		p_bv_->col(nSample) = adapter.getBearingVector(selected_cols_[nSample]);
		if (adapter.isValid(selected_cols_[nSample])){
			p_X_c_->col(nSample) = adapter.getPointCurr(selected_cols_[nSample]);
			nValid++;
		}
	}
	if (nValid == K)
		use_shinji = true;
	//assign the fourth elements for 
	p_X_w_->col(3) = adapter.getPointGlob(selected_cols_[3]);
	p_bv_->col(3)     = adapter.getBearingVector(selected_cols_[3]);

	return use_shinji;
}

template< typename Tp >
void shinji_kneip_ransac(AOPoseAdapter<Tp>& adapter, const Tp dist_thre_3d_, const Tp thre_2d_, int& Iter, Tp confidence = 0.99){
	typedef Matrix<Tp, Dynamic, Dynamic> MatrixX;
	typedef Matrix<Tp, 3, 1> Point3;
	typedef Sophus::SE3<Tp> RT;

	const Tp cos_thr = cos(atan(thre_2d_ / adapter.getFocal()));

	RandomElements<int> re((int)adapter.getNumberCorrespondences());
	const int K = 3;
	MatrixX X_w(3, K + 1), X_c(3, K + 1), bv(3, K + 1);
	Matrix<short, Dynamic, Dynamic> inliers(adapter.getNumberCorrespondences(), 2);

	adapter.setMaxVotes(-1);
	for (int ii = 0; ii < Iter; ii++)	{
		//randomly select K candidates
		RT solution_kneip, solution_shinji;
		vector<RT> v_solutions;
		vector<int> selected_cols;
		re.run(K + 1, &selected_cols);

		if (assign_sample<Tp>(adapter, selected_cols, &X_w, &X_c, &bv)){
			solution_shinji = shinji<Tp>(X_w, X_c, K);
			v_solutions.push_back(solution_shinji);
		}

		if (kneip<Tp>(X_w, bv, &solution_kneip)){
			v_solutions.push_back(solution_kneip);
		}

		for (typename vector<RT>::iterator itr = v_solutions.begin(); itr != v_solutions.end(); ++itr) {
			//collect votes
			int votes = 0;
			inliers.setZero();
			Point3 eivE; Point3 pc; Tp cos_a;

			for (int c = 0; c < adapter.getNumberCorrespondences(); c++) {
				// voted by 3-3 correspondences
				if (adapter.isValid(c)){
					eivE = adapter.getPointCurr(c) - (itr->so3().template cast<Tp>() * adapter.getPointGlob(c) + itr->translation().template cast<Tp>());
					if (eivE.norm() < dist_thre_3d_){
						inliers(c, 1) = 1;
						votes++;
					}
				}
				// voted by 2-3 correspondences
				pc = itr->so3().template cast<Tp>() * adapter.getPointGlob(c) + itr->translation().template cast<Tp>();// transform pw into pc
				pc = pc / pc.norm(); //normalize pc

				//compute the score
				cos_a = pc.transpose() * adapter.getBearingVector(c);
				if (cos_a > cos_thr){
					inliers(c, 0) = 1;
					votes++;
				}
			}
			//cout << endl;

			if (votes > adapter.getMaxVotes()){
				adapter.setMaxVotes(votes);
				adapter.setRcw(itr->so3());
				adapter.sett(itr->translation());
				adapter.setInlier(inliers);
				//Iter = RANSACUpdateNumIters(confidence, (POINT_T)(adapter.getNumberCorrespondences() * 2 - votes) / adapter.getNumberCorrespondences() / 2, K, Iter);
			}
		}
	}
	PnPPoseAdapter<Tp>* pAdapter = &adapter;
	pAdapter->cvtInlier();
	adapter.cvtInlier();

	return;
}


#endif
