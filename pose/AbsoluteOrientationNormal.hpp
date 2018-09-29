#ifndef BTL_AO_NORM_POSE_HEADER
#define BTL_AO_NORM_POSE_HEADER

//#include "common/OtherUtil.hpp"
#include <limits>
#include <Eigen/Dense>
#include "NormalAOPoseAdapter.hpp"
#include "AbsoluteOrientation.hpp"

using namespace Eigen;
using namespace std;

template<typename Tp>
Matrix<Tp, 3, 1> find_opt_cc(NormalAOPoseAdapter<Tp>& adapter)
{
	//the R has been fixed, we need to find optimal cc, camera center, given n pairs of 2-3 correspondences
	//Slabaugh, G., Schafer, R., & Livingston, M. (2001). Optimal Ray Intersection For Computing 3D Points From N -View Correspondences.
	typedef Matrix<Tp, 3, 1> V3;
	typedef Matrix<Tp, 3, 3> M3;

	M3 Rwc = adapter.getRcw().inverse().matrix();
	M3 AA; AA.setZero();
	V3 bb; bb.setZero();
	for (int i = 0; i < adapter.getNumberCorrespondences(); i++)
	{
		if (adapter.isInlier23(i)){
			V3 vr_w = Rwc * adapter.getBearingVector(i);
			M3 A;
			A(0,0) = 1 - vr_w(0)*vr_w(0);
			A(1,0) = A(0,1) = - vr_w(0)*vr_w(1);
			A(2,0) = A(0,2) = - vr_w(0)*vr_w(2);
			A(1,1) = 1 - vr_w(1)*vr_w(1);
			A(2,1) = A(1,2) = - vr_w(1)*vr_w(2);
			A(2,2) = 1 - vr_w(2)*vr_w(2);
			V3 b = A * adapter.getPointGlob(i);
			AA += A;
			bb += b;
		}
	}
	V3 c_w;
	if (fabs(AA.determinant()) < Tp(0.0001))
		c_w = V3(numeric_limits<Tp>::quiet_NaN(), numeric_limits<Tp>::quiet_NaN(), numeric_limits<Tp>::quiet_NaN());
	else
		c_w = AA.jacobiSvd(ComputeFullU | ComputeFullV).solve(bb);
	return c_w;
}

template< typename Tp >
bool assign_sample(const NormalAOPoseAdapter<Tp>& adapter, 
	const vector<int>& selected_cols_, 
	Matrix<Tp, Dynamic, Dynamic>* p_X_w_, Matrix<Tp, Dynamic, Dynamic>* p_N_w_, 
	Matrix<Tp, Dynamic, Dynamic>* p_X_c_, Matrix<Tp, Dynamic, Dynamic>* p_N_c_, Matrix<Tp, Dynamic, Dynamic>* p_bv_){
	
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

template<typename Tp>
void nl_2p( const Matrix<Tp,3,1>& pt1_c, const Matrix<Tp,3,1>& nl1_c, const Matrix<Tp,3,1>& pt2_c, 
			const Matrix<Tp,3,1>& pt1_w, const Matrix<Tp,3,1>& nl1_w, const Matrix<Tp,3,1>& pt2_w, 
			Sophus::SE3<Tp>* p_solution){
	//Drost, B., Ulrich, M., Navab, N., & Ilic, S. (2010). Model globally, match locally: Efficient and robust 3D object recognition. In CVPR (pp. 998?005). Ieee. http://doi.org/10.1109/CVPR.2010.5540108
	// typedef Matrix<Tp, Dynamic, Dynamic> MatrixX;
	typedef Matrix<Tp, 3, 1> V3;
	typedef Sophus::SO3<Tp> ROTATION;
	// typedef Sophus::SE3<Tp> RT;

	V3 c_w = pt1_w; // c_w is the origin of coordinate g w.r.t. world

	Tp alpha = acos(nl1_w(0)); // rotation nl1_c to x axis (1,0,0)
	V3 axis( 0, nl1_w(2), -nl1_w(1)); //rotation axis between nl1_c to x axis (1,0,0) i.e. cross( nl1_w, x );
	axis.normalize();

	//verify quaternion and rotation matrix
	Quaternion<Tp> q_g_f_w(AngleAxis<Tp>(alpha, axis));
	//cout << q_g_f_w << endl;
	ROTATION R_g_f_w(q_g_f_w);
	//cout << R_g_f_w << endl;

	// V3 nl_x = R_g_f_w * nl1_w;
	axis.normalize();

	V3 c_c = pt1_c;
	Tp beta = acos(nl1_c(0)); //rotation nl1_w to x axis (1,0,0)
	V3 axis2( 0, nl1_c(2), -nl1_c(1) ); //rotation axis between nl1_m to x axis (1,0,0) i.e. cross( nl1_w, x );
	axis2.normalize();

	Quaternion<Tp> q_gp_f_c(AngleAxis<Tp>(beta, axis2));
	//cout << q_gp_f_c << endl;
	ROTATION R_gp_f_c(q_gp_f_c);
	//cout << R_gp_f_c << endl;
	//{
	//	Vector3 nl_x = R_gp_f_c * nl1_c;
	//	print<T, Vector3>(nl_x);
	//}

	V3 pt2_g = R_g_f_w * (pt2_w - c_w); pt2_g(0) = Tp(0);  pt2_g.normalize();
	V3 pt2_gp = R_gp_f_c * (pt2_c - c_c); pt2_gp(0) = Tp(0);  pt2_gp.normalize();

	Tp gamma = acos(pt2_g.dot(pt2_gp)); //rotate pt2_g to pt2_gp;
	V3 axis3(1,0,0); 

	Quaternion< Tp > q_gp_f_g(AngleAxis<Tp>(gamma, axis3));
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


template< typename Tp > /*Matrix<float,-1,-1,0,-1,-1> = MatrixXf*/
Sophus::SE3<Tp>  nl_shinji_ls(const Matrix<Tp, Dynamic, Dynamic> & Xw_, const Matrix<Tp, Dynamic, Dynamic>&  Xc_,
	const Matrix<Tp, Dynamic, Dynamic> & Nw_, const Matrix<Tp, Dynamic, Dynamic>&  Nc_, const int K) {
	typedef Sophus::SE3<Tp> RT;

	assert(Xw_.rows() == 3);

	//Compute the centroid of each point set
	Matrix<Tp, 3, 1> Cw(0, 0, 0), Cc(0, 0, 0); //Matrix<float,3,1,0,3,1> = Vector3f
	for (int nCount = 0; nCount < K; nCount++){
		Cw += Xw_.col(nCount);
		Cc += Xc_.col(nCount);
	}
	Cw /= (Tp)K;
	Cc /= (Tp)K;

	// transform coordinate
	Matrix<Tp, 3, 1> Aw, Ac;
	Matrix<Tp, 3, 3> M; M.setZero();
	Matrix<Tp, 3, 3> N;
	Tp sigma_w_sqr = 0;
	for (int nC = 0; nC < K; nC++){
		Aw = Xw_.col(nC) - Cw; sigma_w_sqr += Aw.squaredNorm();
		Ac = Xc_.col(nC) - Cc; 
		N = Ac * Aw.transpose();
		M += N;
	}

	M /= (Tp)K;
	sigma_w_sqr /= (Tp)K;

	Matrix<Tp, 3, 3> M_n; M_n.setZero();
	for (int nC = 0; nC < K; nC++){
		//pure imaginary Shortcuts
		Aw = Nw_.col(nC);
		Ac = Nc_.col(nC);
		N = Ac * Aw.transpose();
		M_n += (sigma_w_sqr*N);
	}

	M_n /= (Tp)K;
	M += M_n;

	JacobiSVD<Matrix<Tp, 3,3> > svd(M, ComputeFullU | ComputeFullV);
	//[U S V]=svd(M);
	//R=U*V';
	Matrix<Tp, 3, 3> U = svd.matrixU();
	Matrix<Tp, 3, 3> V = svd.matrixV();
	Matrix<Tp, 3, 1> D = svd.singularValues();
	Sophus::SO3<Tp> R_tmp;
	Matrix<Tp, 3, 1> TMP = U*V.transpose();
	Tp d = TMP.determinant();
	if (d < 0) {
		Matrix<Tp, 3, 3> I = Matrix<Tp, 3, 3>::Identity(); I(2, 2) = -1; D(2) *= -1;
		R_tmp = Sophus::SO3<Tp>(U*I*V.transpose());
	}
	else{
		R_tmp = Sophus::SO3<Tp>(U*V.transpose());
	}
	//T=ccent'-R*wcent';
	Matrix< Tp, 3, 1> T_tmp = Cc - R_tmp * Cw;

	RT solution = RT( R_tmp, T_tmp) ;

	//T tr = D.sum();
	//T dE_sqr = sigma_c_sqr - tr*tr / sigma_w_sqr;
	//PRINT(dE_sqr);
	return solution; // dE_sqr;
}

template< typename Tp > /*Eigen::Matrix<float,-1,-1,0,-1,-1> = Eigen::MatrixXf*/
void nl_kneip_ransac(NormalAOPoseAdapter<Tp>& adapter, const Tp thre_2d_, const Tp nl_thre, 
	int& Iter, Tp confidence = 0.99){
	typedef Matrix<Tp, Dynamic, Dynamic> MatrixX;
	typedef Matrix<Tp, 3, 1> Point3;
	typedef Sophus::SE3<Tp> RT;

	Tp cos_thr = cos(atan(thre_2d_ / adapter.getFocal()));
	Tp cos_nl_thre = cos(nl_thre);

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

		assign_sample<Tp>(adapter, selected_cols, &Xw, &Nw, &Xc, &Nc, &bv);
		if (!kneip<Tp>(Xw, bv, &solution_kneip))	continue;

		//collect votes
		int votes_kneip = 0;
		inliers_kneip.setZero();
		Point3 eivE; Point3 pc; Tp cos_a;
		for (int c = 0; c < adapter.getNumberCorrespondences(); c++) {
			if (adapter.isValid(c)){
				//with normal data
				Tp cos_alpha = adapter.getNormalCurr(c).dot(solution_kneip.so3() * adapter.getNormalGlob(c));
				if (cos_alpha > cos_nl_thre){
					inliers_kneip(c, 2) = 1;
					votes_kneip++;
				}
			}
			//with 2d
			pc = solution_kneip.so3() * adapter.getPointGlob(c) + solution_kneip.translation();// transform pw into pc
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
			Iter = RANSACUpdateNumIters(confidence, (Tp)(adapter.getNumberCorrespondences() * 2 - votes_kneip) / adapter.getNumberCorrespondences() / 2, K, Iter);
		}
	}
	PnPPoseAdapter<Tp>* pAdapter = &adapter;
	pAdapter->cvtInlier();
	adapter.cvtInlier();

	return;
}

template< typename Tp > /*Eigen::Matrix<float,-1,-1,0,-1,-1> = Eigen::MatrixXf*/
void nl_shinji_ransac(NormalAOPoseAdapter<Tp>& adapter, const Tp thre_3d_, const Tp nl_thre, 
	int& Iter, Tp confidence = 0.99){
	// eimXc_ = R * eimXw_ + T; //R and t is defined in world and transform a point in world to local
	typedef Matrix<Tp, Dynamic, Dynamic> MatrixX;
	typedef Matrix<Tp, 3, 1> Point3;
	typedef Sophus::SE3<Tp> RT;

	Tp cos_nl_thre = cos(nl_thre);

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

		if (assign_sample<Tp>(adapter, selected_cols, &Xw, &Nw, &Xc, &Nc, &bv)){
			solution_shinji = shinji<Tp>(Xw, Xc, K);
			v_solutions.push_back(solution_shinji);
		}

		nl_2p<Tp>(Xc.col(0), Nc.col(0), Xc.col(1), Xw.col(0), Nw.col(0), Xw.col(1), &solution_nl);
		v_solutions.push_back(solution_nl);
		for (typename vector<RT>::iterator itr = v_solutions.begin(); itr != v_solutions.end(); ++itr) {
			//collect votes
			int votes = 0;
			inliers.setZero();
			Point3 eivE; Point3 pc; 
			for (int c = 0; c < adapter.getNumberCorrespondences(); c++) {
				if (adapter.isValid(c)){
					//voted by N-N correspondences
					Tp cos_alpha = adapter.getNormalCurr(c).dot(itr->so3() * adapter.getNormalGlob(c));
					if (cos_alpha > cos_nl_thre){
						inliers(c, 2) = 1;
						votes++;
					}
					//voted by 3-3 correspondences
					eivE = adapter.getPointCurr(c) - (itr->so3() * adapter.getPointGlob(c) + itr->translation());
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
				Iter = RANSACUpdateNumIters(confidence, (Tp)(adapter.getNumberCorrespondences() * 2 - votes) / adapter.getNumberCorrespondences() / 2, K, Iter);
			}
		}
	}
	AOPoseAdapter<Tp>* pAdapter = &adapter;
	pAdapter->cvtInlier();
	adapter.cvtInlier();
	return;
}

template< typename Tp > 
void nl_shinji_kneip_ransac(NormalAOPoseAdapter<Tp>& adapter, 
	const Tp thre_3d_, const Tp thre_2d_, const Tp nl_thre, int& Iter, Tp confidence = 0.99){
	typedef Matrix<Tp, Dynamic, Dynamic> MX;
	typedef Matrix<Tp, 3, 1> P3;
	typedef Sophus::SE3<Tp> RT;

	Tp cos_thr = cos(atan(thre_2d_ / adapter.getFocal()));
	Tp cos_nl_thre = cos(nl_thre);

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

		if (assign_sample<Tp>(adapter, selected_cols, &Xw, &Nw, &Xc, &Nc, &bv)){
			solution_shinji = shinji<Tp>(Xw, Xc, K);
			v_solutions.push_back(solution_shinji);
		}

		if (kneip<Tp>(Xw, bv, &solution_kneip)){
			v_solutions.push_back(solution_kneip);
		}

		nl_2p<Tp>(Xc.col(0), Nc.col(0), Xc.col(1), Xw.col(0), Nw.col(0), Xw.col(1), &solution_nl);
		v_solutions.push_back(solution_nl);

		for (typename vector<RT>::iterator itr = v_solutions.begin(); itr != v_solutions.end(); ++itr) {
			//collect votes
			int votes = 0;
			inliers.setZero();
			P3 eivE; P3 pc; Tp cos_a;
			for (int c = 0; c < adapter.getNumberCorrespondences(); c++) {
				if (adapter.isValid(c)){
					//with normal data
					Tp cos_alpha = adapter.getNormalCurr(c).dot(itr->so3() * adapter.getNormalGlob(c));
					if (cos_alpha > cos_nl_thre){
						inliers(c, 2) = 1;
						votes++;
					}
				
					//with 3d data
					eivE = adapter.getPointCurr(c) - (itr->so3() * adapter.getPointGlob(c) + itr->translation());
					if (eivE.norm() < thre_3d_){
						inliers(c, 1) = 1;
						votes++;
					}
				}
				//with 2d
				pc = itr->so3() * adapter.getPointGlob(c) + itr->translation();// transform pw into pc
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
				Iter = RANSACUpdateNumIters(confidence, (Tp)(adapter.getNumberCorrespondences() * 3 - votes) / adapter.getNumberCorrespondences() / 3, K, Iter);
			}
		}//for(vector<RT>::iterator itr = v_solutions.begin() ...
	}//for(int ii = 0; ii < Iter; ii++)
	PnPPoseAdapter<Tp>* pAdapterPnP = &adapter;
	pAdapterPnP->cvtInlier();
	AOPoseAdapter<Tp>* pAdapterAO = &adapter;
	pAdapterAO->cvtInlier();
	adapter.cvtInlier();
	return;
}

template< typename Tp > 
void nl_shinji_kneip_ls(NormalAOPoseAdapter<Tp>& adapter)
{
	// typedef Matrix<Tp, Dynamic, Dynamic> MatrixX;
	typedef Matrix<Tp, 3, 1> V3;
	typedef Matrix<Tp, 3, 3> M3;
	// typedef Matrix<Tp, 3, 4> RT;
	if(adapter.getMaxVotes() == 0) return;

	//Compute the centroid of each point set
	V3 Cw(0, 0, 0), Cc(0, 0, 0); int N = 0; Tp TV = 0;
	for (int nCount = 0; nCount < adapter.getNumberCorrespondences(); nCount++){
		if (adapter.isInlier33(nCount)){
			Tp v = adapter.weight33(nCount);
			Cw += (v * adapter.getPointGlob(nCount));
			Cc += (v * adapter.getPointCurr(nCount));
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
	int M = 0; Tp TL = 0;
	int K = 0; Tp TW = 0;

	V3 c_opt = adapter.getRcw().inverse() * (-adapter.gettw()); //camera centre in WRS
	Sophus::SO3<Tp> R_opt;
	for (int ii = 0; ii < 3; ii++)
	{
		Tp sigma_w_sqr = 0.;
		for (int nC = 0; nC < adapter.getNumberCorrespondences(); nC++){
			if (adapter.isInlier23(nC)){
				Tp w = adapter.weight23(nC);
				Aw = adapter.getPointGlob(nC) - c_opt; Aw.normalize();
				Ac = adapter.getBearingVector(nC);
				M23 += (w * Ac * Aw.transpose());
				TW += w;  K++;
			}
			if (adapter.isInlier33(nC)){
				Tp v = adapter.weight33(nC);
				Aw = adapter.getPointGlob(nC) - Cw;
				Ac = adapter.getPointCurr(nC) - Cc; sigma_w_sqr += (v*Ac.squaredNorm());
				M33 += (v * Ac * Aw.transpose());
			}
			if (adapter.isInlierNN(nC)){
				Tp lambda = adapter.weightNN(nC);
				Aw = adapter.getNormalGlob(nC);
				Ac = adapter.getNormalCurr(nC);
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
		// V3 D = svd.singularValues();
		M3 TMP = U*V.transpose();
		Tp d = TMP.determinant();
		if (d < 0) {
			M3 I = M3::Identity(); I(2, 2) = -1; //D(2) *= -1;
			R_opt = Sophus::SO3<Tp>(U*I*V.transpose());
		}
		else{
			R_opt = Sophus::SO3<Tp>(U*V.transpose());
		}
		//T=ccent'-R*wcent';
		V3 c = Cw - R_opt.inverse() * Cc;
		V3 cp = find_opt_cc<Tp>(adapter);
		if (N > 2 ){
			if (cp[0] == cp[0])
				c_opt = Tp(K) / (K + N)*cp + Tp(N) / (K + N)*c;
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
