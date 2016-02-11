#ifndef BTL_AO_NORM_POSE_HEADER
#define BTL_AO_NORM_POSE_HEADER

#include "common/OtherUtil.hpp"
#include <limits>
#include <Eigen/Dense>
#include "NormalAOPoseAdapter.hpp"
#include "AbsoluteOrientation.hpp"

using namespace Eigen;
using namespace std;

template<typename POSE_T, typename POINT_T>
Matrix<POSE_T, 3, 1> find_opt_cc(NormalAOPoseAdapter<POSE_T, POINT_T>& adapter)
{
	//the R has been fixed, we need to find optimal cc, camera center, given n pairs of 2-3 correspondences
	//Slabaugh, G., Schafer, R., & Livingston, M. (2001). Optimal Ray Intersection For Computing 3D Points From N -View Correspondences.
	typedef Matrix<POSE_T, 3, 1> V3;
	typedef Matrix<POSE_T, 3, 3> M3;

	M3 Rwc = adapter.getRcw().inverse().matrix();
	M3 AA; AA.setZero();
	V3 bb; bb.setZero();
	for (int i = 0; i < adapter.getNumberCorrespondences(); i++)
	{
		if (adapter.isInlier23(i)){
			V3 vr_w = Rwc * adapter.getBearingVector(i).template cast<POSE_T>();
			M3 A;
			A(0,0) = 1 - vr_w(0)*vr_w(0);
			A(1,0) = A(0,1) = - vr_w(0)*vr_w(1);
			A(2,0) = A(0,2) = - vr_w(0)*vr_w(2);
			A(1,1) = 1 - vr_w(1)*vr_w(1);
			A(2,1) = A(1,2) = - vr_w(1)*vr_w(2);
			A(2,2) = 1 - vr_w(2)*vr_w(2);
			V3 b = A * adapter.getPointGlob(i).template cast<POSE_T>();
			AA += A;
			bb += b;
		}
	}
	V3 c_w;
	if (fabs(AA.determinant()) < POSE_T(0.0001))
		c_w = V3(numeric_limits<POSE_T>::quiet_NaN(), numeric_limits<POSE_T>::quiet_NaN(), numeric_limits<POSE_T>::quiet_NaN());
	else
		c_w = AA.jacobiSvd(ComputeFullU | ComputeFullV).solve(bb);
	return c_w;
}

template< typename POSE_T, typename POINT_T >
bool assign_sample(const NormalAOPoseAdapter<POSE_T, POINT_T>& adapter, 
	const vector<int>& selected_cols_, 
	Matrix<POINT_T, Dynamic, Dynamic>* p_X_w_, Matrix<POINT_T, Dynamic, Dynamic>* p_N_w_, 
	Matrix<POINT_T, Dynamic, Dynamic>* p_X_c_, Matrix<POINT_T, Dynamic, Dynamic>* p_N_c_, Matrix<POINT_T, Dynamic, Dynamic>* p_bv_){
	
	int K = (int)selected_cols_.size() - 1;
	bool use_shinji = false;
	int nValid = 0;
	for (int nSample = 0; nSample < K; nSample++) {
		p_X_w_->col(nSample) = adapter.getPointGlob(selected_cols_[nSample]);
		p_N_w_->col(nSample) = adapter.getNormalGlob(selected_cols_[nSample]);
		p_bv_->col(nSample) = adapter.getBearingVector(selected_cols_[nSample]);
		if (adapter.isValid(selected_cols_[nSample])){
			p_X_c_->col(nSample) = adapter.getPointCurr(selected_cols_[nSample]);
			p_N_c_->col(nSample) = adapter.getNormalCurr(selected_cols_[nSample]);
			nValid++;
		}
	}
	if (nValid == K)
		use_shinji = true;
	//assign the fourth elements for 
	p_X_w_->col(3) = adapter.getPointGlob(selected_cols_[3]);
	p_N_w_->col(3) = adapter.getNormalGlob(selected_cols_[3]);
	p_bv_->col(3) = adapter.getBearingVector(selected_cols_[3]);

	return use_shinji;
}

template<typename POSE_T, typename POINT_T>
void nl_2p( const Matrix<POINT_T,3,1>& pt1_c, const Matrix<POINT_T,3,1>& nl1_c, const Matrix<POINT_T,3,1>& pt2_c, 
			const Matrix<POINT_T,3,1>& pt1_w, const Matrix<POINT_T,3,1>& nl1_w, const Matrix<POINT_T,3,1>& pt2_w, 
			SE3Group<POSE_T>* p_solution){
	//Drost, B., Ulrich, M., Navab, N., & Ilic, S. (2010). Model globally, match locally: Efficient and robust 3D object recognition. In CVPR (pp. 998?005). Ieee. http://doi.org/10.1109/CVPR.2010.5540108
	typedef Matrix<POINT_T, Dynamic, Dynamic> MatrixX;
	typedef Matrix<POSE_T, 3, 1> V3;
	typedef SO3Group<POSE_T> ROTATION;
	typedef SE3Group<POSE_T> RT;

	V3 c_w = pt1_w.template cast<POSE_T>(); // c_w is the origin of coordinate g w.r.t. world

	POSE_T alpha = acos(nl1_w(0)); // rotation nl1_c to x axis (1,0,0)
	V3 axis( 0, nl1_w(2), -nl1_w(1)); //rotation axis between nl1_c to x axis (1,0,0) i.e. cross( nl1_w, x );
	axis.normalized();

	//verify quaternion and rotation matrix
	Quaternion<POSE_T> q_g_f_w(AngleAxis<POSE_T>(alpha, axis));
	//cout << q_g_f_w << endl;
	ROTATION R_g_f_w(q_g_f_w);
	//cout << R_g_f_w << endl;

	V3 nl_x = R_g_f_w * nl1_w.template cast<POSE_T>();
	axis.normalized();

	V3 c_c = pt1_c.template cast<POSE_T>();
	POSE_T beta = acos(nl1_c(0)); //rotation nl1_w to x axis (1,0,0)
	V3 axis2( 0, nl1_c(2), -nl1_c(1) ); //rotation axis between nl1_m to x axis (1,0,0) i.e. cross( nl1_w, x );
	axis2.normalized();

	Quaternion<POSE_T> q_gp_f_c(AngleAxis<POSE_T>(beta, axis2));
	//cout << q_gp_f_c << endl;
	ROTATION R_gp_f_c(q_gp_f_c);
	//cout << R_gp_f_c << endl;
	//{
	//	Vector3 nl_x = R_gp_f_c * nl1_c;
	//	print<T, Vector3>(nl_x);
	//}

	V3 pt2_g = R_g_f_w * (pt2_w.template cast<POSE_T>() - c_w); pt2_g(0) = POINT_T(0);  pt2_g.normalized();
	V3 pt2_gp = R_gp_f_c * (pt2_c.template cast<POSE_T>() - c_c); pt2_gp(0) = POINT_T(0);  pt2_gp.normalized();

	POSE_T gamma = acos(pt2_g.dot(pt2_gp)); //rotate pt2_g to pt2_gp;
	V3 axis3(1,0,0); 

	Quaternion< POSE_T > q_gp_f_g(AngleAxis<POSE_T>(gamma, axis3));
	//cout << q_gp_f_g << endl;
	ROTATION R_gp_f_g ( q_gp_f_g );
	//cout << R_gp_f_g << endl;

	ROTATION R_c_f_gp = R_gp_f_c.inverse();
	p_solution->so3() = R_c_f_gp * R_gp_f_g * R_g_f_w;
	//{
	//	T3 pt = *R_cfw * (pt2_w - c_w) + c_c;
	//	cout << norm<T, T3>( pt - pt2_c ) << endl;
	//}
	//{
	//	cout << norm<T, T3>(nl1_w) << endl;
	//	cout << norm<T, T3>(nl1_c) << endl;
	//	cout << norm<T, T3>(*R_cfw * nl1_w) << endl;
	//	cout << norm<T, T3>(nl1_c - *R_cfw * nl1_w) << endl;
	//}
	p_solution->translation() = c_c - p_solution->so3() * c_w;

	return;
}


template< typename POSE_T, typename POINT_T > /*Matrix<float,-1,-1,0,-1,-1> = MatrixXf*/
SE3Group<POSE_T>  nl_shinji_ls(const Matrix<POINT_T, Dynamic, Dynamic> & Xw_, const Matrix<POINT_T, Dynamic, Dynamic>&  Xc_,
	const Matrix<POINT_T, Dynamic, Dynamic> & Nw_, const Matrix<POINT_T, Dynamic, Dynamic>&  Nc_, const int K) {
	typedef SE3Group<POSE_T> RT;

	assert(Xw_.rows() == 3);

	//Compute the centroid of each point set
	Matrix<POINT_T, 3, 1> Cw(0, 0, 0), Cc(0, 0, 0); //Matrix<float,3,1,0,3,1> = Vector3f
	for (int nCount = 0; nCount < K; nCount++){
		Cw += Xw_.col(nCount);
		Cc += Xc_.col(nCount);
	}
	Cw /= (POINT_T)K;
	Cc /= (POINT_T)K;

	// transform coordinate
	Matrix<POSE_T, 3, 1> Aw, Ac;
	Matrix<POSE_T, 3, 3> M; M.setZero();
	Matrix<POSE_T, 3, 3> N;
	POSE_T sigma_w_sqr = 0;
	for (int nC = 0; nC < K; nC++){
		Aw = Xw_.col(nC) - Cw; sigma_w_sqr += Aw.squaredNorm();
		Ac = Xc_.col(nC) - Cc; 
		N = Ac * Aw.transpose();
		M += N;
	}

	M /= (POSE_T)K;
	sigma_w_sqr /= (POSE_T)K;

	Matrix<POSE_T, 3, 3> M_n; M_n.setZero();
	for (int nC = 0; nC < K; nC++){
		//pure imaginary Shortcuts
		Aw = Nw_.col(nC);
		Ac = Nc_.col(nC);
		N = Ac * Aw.transpose();
		M_n += (sigma_w_sqr*N);
	}

	M_n /= (POSE_T)K;
	M += M_n;

	JacobiSVD<Matrix<POSE_T, 3,3> > svd(M, ComputeFullU | ComputeFullV);
	//[U S V]=svd(M);
	//R=U*V';
	Matrix<POSE_T, 3, 3> U = svd.matrixU();
	Matrix<POSE_T, 3, 3> V = svd.matrixV();
	Matrix<POSE_T, 3, 1> D = svd.singularValues();
	SO3Group<POSE_T> R_tmp;
	if (M.determinant() < 0){
		Matrix<POSE_T, 3, 3> I = Matrix<POSE_T, 3, 3>::Identity(); I(2, 2) = -1; D(2) *= -1;
		R_tmp = SO3Group<POSE_T>(U*I*V.transpose());
	}
	else{
		R_tmp = SO3Group<POSE_T>(U*V.transpose());
	}
	//T=ccent'-R*wcent';
	Matrix< POSE_T, 3, 1> T_tmp = Cc - R_tmp * Cw;

	RT solution = RT( R_tmp, T_tmp) ;

	//T tr = D.sum();
	//T dE_sqr = sigma_c_sqr - tr*tr / sigma_w_sqr;
	//PRINT(dE_sqr);
	return solution; // dE_sqr;
}

template< typename POSE_T, typename POINT_T > /*Eigen::Matrix<float,-1,-1,0,-1,-1> = Eigen::MatrixXf*/
void nl_kneip_ransac(NormalAOPoseAdapter<POSE_T, POINT_T>& adapter, const POINT_T thre_2d_, const POINT_T nl_thre, 
	int& Iter, POINT_T confidence = 0.99){
	typedef Matrix<POINT_T, Dynamic, Dynamic> MatrixX;
	typedef Matrix<POINT_T, 3, 1> Point3;
	typedef SE3Group<POSE_T> RT;

	POSE_T cos_thr = cos(atan(thre_2d_ / adapter.getFocal()));
	POINT_T cos_nl_thre = cos(nl_thre);

	RandomElements<int> re((int)adapter.getNumberCorrespondences());
	const int K = 3;
	MatrixX Xw(3, K + 1), Xc(3, K + 1), bv(3, K + 1);
	MatrixX Nw(3, K + 1), Nc(3, K + 1);
	Matrix<short, Dynamic, Dynamic> inliers_kneip(adapter.getNumberCorrespondences(), 3); 

	adapter.setMaxVotes(-1);
	for (int ii = 0; ii < Iter; ii++)	{
		//randomly select K candidates
		RT solution_kneip;
		vector<int> selected_cols;
		re.run(K + 1, &selected_cols);

		assign_sample<POSE_T, POINT_T>(adapter, selected_cols, &Xw, &Nw, &Xc, &Nc, &bv);
		if (!kneip<POSE_T, POINT_T>(Xw, bv, &solution_kneip))	continue;

		//collect votes
		int votes_kneip = 0;
		inliers_kneip.setZero();
		Point3 eivE; Point3 pc; POINT_T cos_a;
		for (int c = 0; c < adapter.getNumberCorrespondences(); c++) {
			if (adapter.isValid(c)){
				//with normal data
				POINT_T cos_alpha = adapter.getNormalCurr(c).dot(solution_kneip.so3().template cast<POINT_T>() * adapter.getNormalGlob(c));
				if (cos_alpha > cos_nl_thre){
					inliers_kneip(c, 2) = 1;
					votes_kneip++;
				}
			}
			//with 2d
			pc = solution_kneip.so3().template cast<POINT_T>() * adapter.getPointGlob(c) + solution_kneip.translation().template cast<POINT_T>();// transform pw into pc
			pc = pc / pc.norm(); //normalize pc

			//compute the score
			cos_a = pc.dot( adapter.getBearingVector(c) );
			if (cos_a > cos_thr){
				inliers_kneip(c, 0) = 1;
				votes_kneip++;
			}
		}
		//cout << endl;

		if ( votes_kneip > adapter.getMaxVotes() ){
			assert(votes_kneip == inliers_kneip.sum());
			adapter.setMaxVotes(votes_kneip);
			adapter.setRcw(solution_kneip.so3());
			adapter.sett(solution_kneip.translation());
			adapter.setInlier(inliers_kneip);

			//cout << inliers_kneip.inverse() << endl << endl;
			//adapter.printInlier();
			//Iter = RANSACUpdateNumIters(confidence, (POINT_T)(adapter.getNumberCorrespondences() * 2 - votes_kneip) / adapter.getNumberCorrespondences() / 2, K, Iter);
		}
	}
	PnPPoseAdapter<POSE_T, POINT_T>* pAdapter = &adapter;
	pAdapter->cvtInlier();
	adapter.cvtInlier();

	return;
}

template< typename POSE_T, typename POINT_T > /*Eigen::Matrix<float,-1,-1,0,-1,-1> = Eigen::MatrixXf*/
void nl_shinji_ransac(NormalAOPoseAdapter<POSE_T, POINT_T>& adapter, const POINT_T thre_3d_, const POINT_T nl_thre, 
	int& Iter, POINT_T confidence = 0.99){
	// eimXc_ = R * eimXw_ + T; //R and t is defined in world and transform a point in world to local
	typedef Matrix<POINT_T, Dynamic, Dynamic> MatrixX;
	typedef Matrix<POINT_T, 3, 1> Point3;
	typedef SE3Group<POSE_T> RT;

	POINT_T cos_nl_thre = cos(nl_thre);

	RandomElements<int> re((int)adapter.getNumberCorrespondences());
	const int K = 3;
	MatrixX Xw(3, K + 1), Xc(3, K + 1), bv(3, K + 1);
	MatrixX Nw(3, K + 1), Nc(3, K + 1);
	Matrix<short, Dynamic, Dynamic> inliers(adapter.getNumberCorrespondences(), 3);

	adapter.setMaxVotes(-1);
	for (int ii = 0; ii < Iter; ii++)	{
		//randomly select K candidates
		RT solution_shinji, solution_nl;
		vector<int> selected_cols;
		re.run(K + 1, &selected_cols);
		vector<RT> v_solutions;

		if (assign_sample<POSE_T, POINT_T>(adapter, selected_cols, &Xw, &Nw, &Xc, &Nc, &bv)){
			solution_shinji = shinji<POSE_T, POINT_T>(Xw, Xc, K);
			v_solutions.push_back(solution_shinji);
		}

		nl_2p<POSE_T, POINT_T>(Xc.col(0), Nc.col(0), Xc.col(1), Xw.col(0), Nw.col(0), Xw.col(1), &solution_nl);
		v_solutions.push_back(solution_nl);
		for (typename vector<RT>::iterator itr = v_solutions.begin(); itr != v_solutions.end(); ++itr) {
			//collect votes
			int votes = 0;
			inliers.setZero();
			Point3 eivE; Point3 pc; POINT_T score;
			for (int c = 0; c < adapter.getNumberCorrespondences(); c++) {
				if (adapter.isValid(c)){
					//voted by N-N correspondences
					POINT_T cos_alpha = adapter.getNormalCurr(c).dot(itr->so3().template cast<POINT_T>() * adapter.getNormalGlob(c));
					if (cos_alpha > cos_nl_thre){
						inliers(c, 2) = 1;
						votes++;
					}
					//voted by 3-3 correspondences
					eivE = adapter.getPointCurr(c) - (itr->so3().template cast<POINT_T>() * adapter.getPointGlob(c) + itr->translation().template cast<POINT_T>());
					if (eivE.norm() < thre_3d_){
						inliers(c, 1) = 1;
						votes++;
					}
				}
			}
			//cout << endl;
			if (votes > adapter.getMaxVotes() ){
				assert(votes == inliers.sum());
				adapter.setMaxVotes(votes);
				adapter.setRcw(itr->so3());
				adapter.sett(itr->translation());
				adapter.setInlier(inliers);
				//adapter.printInlier();
				//Iter = RANSACUpdateNumIters(confidence, (POINT_T)(adapter.getNumberCorrespondences() * 2 - votes) / adapter.getNumberCorrespondences() / 2, K, Iter);
			}
		}
	}
	AOPoseAdapter<POSE_T, POINT_T>* pAdapter = &adapter;
	pAdapter->cvtInlier();
	adapter.cvtInlier();
	return;
}

template< typename POSE_T, typename POINT_T > 
void nl_shinji_kneip_ransac(NormalAOPoseAdapter<POSE_T, POINT_T>& adapter, 
	const POINT_T thre_3d_, const POINT_T thre_2d_, const POINT_T nl_thre, int& Iter, POINT_T confidence = 0.99){
	typedef Matrix<POINT_T, Dynamic, Dynamic> MX;
	typedef Matrix<POINT_T, 3, 1> P3;
	typedef SE3Group<POSE_T> RT;

	POSE_T cos_thr = cos(atan(thre_2d_ / adapter.getFocal()));
	POINT_T cos_nl_thre = cos(nl_thre);

	RandomElements<int> re((int)adapter.getNumberCorrespondences());
	const int K = 3;
	MX Xw(3, K + 1), Xc(3, K + 1), bv(3, K + 1);
	MX Nw(3, K + 1), Nc(3, K + 1);
	Matrix<short, Dynamic, Dynamic> inliers(adapter.getNumberCorrespondences(), 3);

	adapter.setMaxVotes(-1);
	for (int ii = 0; ii < Iter; ii++)	{
		//randomly select K candidates
		RT solution_kneip, solution_shinji, solution_nl;
		vector<RT> v_solutions;
		vector<int> selected_cols;
		re.run(K + 1, &selected_cols);

		if (assign_sample<POSE_T, POINT_T>(adapter, selected_cols, &Xw, &Nw, &Xc, &Nc, &bv)){
			solution_shinji = shinji<POSE_T, POINT_T>(Xw, Xc, K);
			v_solutions.push_back(solution_shinji);
		}

		if (kneip<POSE_T, POINT_T>(Xw, bv, &solution_kneip)){
			v_solutions.push_back(solution_kneip);
		}

		nl_2p<POSE_T, POINT_T>(Xc.col(0), Nc.col(0), Xc.col(1), Xw.col(0), Nw.col(0), Xw.col(1), &solution_nl);
		v_solutions.push_back(solution_nl);

		for (typename vector<RT>::iterator itr = v_solutions.begin(); itr != v_solutions.end(); ++itr) {
			//collect votes
			int votes = 0;
			inliers.setZero();
			P3 eivE; P3 pc; POINT_T cos_a;
			for (int c = 0; c < adapter.getNumberCorrespondences(); c++) {
				if (adapter.isValid(c)){
					//with normal data
					POINT_T cos_alpha = adapter.getNormalCurr(c).dot(itr->so3().template cast<POINT_T>() * adapter.getNormalGlob(c));
					if (cos_alpha > cos_nl_thre){
						inliers(c, 2) = 1;
						votes++;
					}
				
					//with 3d data
					eivE = adapter.getPointCurr(c) - (itr->so3().template cast<POINT_T>() * adapter.getPointGlob(c) + itr->translation().template cast<POINT_T>());
					if (eivE.norm() < thre_3d_){
						inliers(c, 1) = 1;
						votes++;
					}
				}
				//with 2d
				pc = itr->so3().template cast<POINT_T>() * adapter.getPointGlob(c) + itr->translation().template cast<POINT_T>();// transform pw into pc
				pc = pc / pc.norm(); //normalize pc

				//compute the score
				cos_a = pc.dot( adapter.getBearingVector(c) );
				if (cos_a > cos_thr){
					inliers(c, 0) = 1;
					votes++;
				}
			}
			//cout << endl;

			if (votes > adapter.getMaxVotes() ){
				assert(votes == inliers.sum());
				adapter.setMaxVotes(votes);
				adapter.setRcw(itr->so3());
				adapter.sett(itr->translation());
				adapter.setInlier(inliers);

				//cout << inliers.inverse() << endl << endl;
				//adapter.printInlier();
				//Iter = RANSACUpdateNumIters(confidence, (POINT_T)(adapter.getNumberCorrespondences() * 3 - votes) / adapter.getNumberCorrespondences() / 3, K, Iter);
			}
		}//for(vector<RT>::iterator itr = v_solutions.begin() ...
	}//for(int ii = 0; ii < Iter; ii++)
	PnPPoseAdapter<POSE_T, POINT_T>* pAdapterPnP = &adapter;
	pAdapterPnP->cvtInlier();
	AOPoseAdapter<POSE_T, POINT_T>* pAdapterAO = &adapter;
	pAdapterAO->cvtInlier();
	adapter.cvtInlier();
	return;
}

template< typename POSE_T, typename POINT_T > 
void nl_shinji_kneip_ls(NormalAOPoseAdapter<POSE_T, POINT_T>& adapter)
{
	typedef Matrix<POINT_T, Dynamic, Dynamic> MatrixX;
	typedef Matrix<POSE_T, 3, 1> V3;
	typedef Matrix<POSE_T, 3, 3> M3;
	typedef Matrix<POSE_T, 3, 4> RT;
	if(adapter.getMaxVotes() == 0) return;

	//Compute the centroid of each point set
	V3 Cw(0, 0, 0), Cc(0, 0, 0); int N = 0; POSE_T TV = 0;
	for (int nCount = 0; nCount < adapter.getNumberCorrespondences(); nCount++){
		if (adapter.isInlier33(nCount)){
			POINT_T v = adapter.weight33(nCount);
			Cw += (v * adapter.getPointGlob(nCount).template cast<POSE_T>());
			Cc += (v * adapter.getPointCurr(nCount).template cast<POSE_T>());
			TV += v; N++;
		}
	}
	if (N > 2){
		Cw /= TV;
		Cc /= TV;
	}

	// transform coordinate
	V3 Aw, Ac;
	M3 M33; M33.setZero();
	M3 MNN; MNN.setZero();
	M3 M23; M23.setZero();
	int M = 0; POSE_T TL = 0;
	int K = 0; POSE_T TW = 0;

	V3 c_opt = adapter.getRcw().inverse() * (-adapter.gettw()); //camera centre in WRS
	SO3Group<POSE_T> R_opt;
	for (int ii = 0; ii < 3; ii++)
	{
		POSE_T sigma_w_sqr = 0.;
		for (int nC = 0; nC < adapter.getNumberCorrespondences(); nC++){
			if (adapter.isInlier23(nC)){
				POSE_T w = adapter.weight23(nC);
				Aw = adapter.getPointGlob(nC).template cast<POSE_T>() - c_opt; Aw.normalize();
				Ac = adapter.getBearingVector(nC).template cast<POSE_T>();
				M23 += (w * Ac * Aw.transpose());
				TW += w;  K++;
			}
			if (adapter.isInlier33(nC)){
				POINT_T v = adapter.weight33(nC);
				Aw = adapter.getPointGlob(nC).template cast<POSE_T>() - Cw;
				Ac = adapter.getPointCurr(nC).template cast<POSE_T>() - Cc; sigma_w_sqr += (v*Ac.squaredNorm());
				M33 += (v * Ac * Aw.transpose());
			}
			if (adapter.isInlierNN(nC)){
				POINT_T lambda = adapter.weightNN(nC);
				Aw = adapter.getNormalGlob(nC).template cast<POSE_T>();
				Ac = adapter.getNormalCurr(nC).template cast<POSE_T>();
				MNN += (lambda * Ac * Aw.transpose());
				TL += lambda;  M++;
			}
		}
		if (N > 2) { M33 /= TV; sigma_w_sqr /= TV; } else { M33.setZero(); sigma_w_sqr = 1.;}
		if (M > 0) { MNN /= TL; } else { MNN.setZero(); }
		if (K > 0) { M23 /= TW; } else { M23.setZero(); }

		M33 += sigma_w_sqr*( M23 + MNN );

		JacobiSVD< M3 > svd(M33, ComputeFullU | ComputeFullV);
		//[U S V]=svd(M);
		//R=U*V';
		M3 U = svd.matrixU();
		M3 V = svd.matrixV();
		V3 D = svd.singularValues();
		if (M33.determinant() < 0){
			M3 I = M3::Identity(); I(2, 2) = -1; //D(2) *= -1;
			R_opt = SO3Group<POSE_T>(U*I*V.transpose());
		}
		else{
			R_opt = SO3Group<POSE_T>(U*V.transpose());
		}
		//T=ccent'-R*wcent';
		V3 c = Cw - R_opt.inverse() * Cc;
		V3 cp = find_opt_cc<POSE_T,POINT_T>(adapter);
		if (N > 2 ){
			if (cp[0] == cp[0])
				c_opt = POSE_T(K) / (K + N)*cp + POSE_T(N) / (K + N)*c;
			else
				c_opt = c;
		}
		else{
			if (cp[0] == cp[0])
				c_opt = cp;
			else
				break;
		}
	}//for iterations
	adapter.setRcw(R_opt);
	adapter.sett(R_opt*(-c_opt));

	//T tr = D.sum();
	//T dE_sqr = sigma_c_sqr - tr*tr / sigma_w_sqr;
	//PRINT(dE_sqr);
	return;
}



#endif
