
#ifndef BTL_MIN_SOLV_HEADER
#define BTL_MIN_SOLV_HEADER

#include "common/OtherUtil.hpp"
#include <Eigen/Dense>
#include <se3.hpp>

using namespace Eigen;
using namespace cv;

template< typename T > 
void ms(const Matrix< T, 3, 1 >& Aw_, const Matrix< T, 3, 1 >& Bw_,
	const Matrix< T, 3, 1 >& Nw_, const Matrix< T, 3, 1 >& Mw_,
	const Matrix< T, 3, 1 >& Ac_, const Matrix< T, 3, 1 >& Bc_,
	const Matrix< T, 3, 1 >& Nc_, const Matrix< T, 3, 1 >& Mc_,
	Sophus::SO3< T >* pR_cw_, Matrix< T, 3, 1 >* pTw_){
	// the minimal solver to solve the 2 pairs of correspondences
	// where each correspondence contains a 3-D location as well as a surface normal
	// Xc_ = R_cw * Xw_ + t_w; //R_cw and t_w is defined in world and transform a point in world to local

	//1
	Matrix< T, 3, 1 > Cw = (Aw_ + Bw_) / T(2); // centre of AwBw
	Matrix< T, 3, 1 > Cc = (Ac_ + Bc_) / T(2); // centre of AcBc

	//2 estimate the translation
	*pTw_ = Cc - Cw;

	//3 translate 
	Aw_ += *pTw_;
	Bw_ += *pTw_;

	//4 rotate around the axis perpendicular to the plane determined by AwBw and AcBc
	Matrix< T, 3, 1 > ABw = Aw_ - Bw_;
	Matrix< T, 3, 1 > ABc = Ac_ - Bc_;

	Matrix< T, 3, 1 > A1 = ABw.cross(ABc);
	T theta = asin(A1.norm() / ABw.norm() / ABc.norm());
	Matrix< T, 3, 3 > R1 = AngleAxis<T>(A1.normalized(), theta).toRotationMatrix();

	Aw_ = R1*Aw_;
	Bw_ = R1*Bw_;
	Nw_ = R1*Nw_;
	Mw_ = R1*Mw_;

	//5 rotation around axis ABc

	return;
}

template< class T > 
void ev(const Matrix<T, 3, 3>& M_, Matrix<T, 3, 1>* pE_, Matrix<T, 3, 3>* pV_=NULL)
{
	//source http://en.wikipedia.org/wiki/Eigenvalue_algorithm
	T p1 = M_(0, 1)*M_(0, 1) + M_(0, 2)*M_(0, 2) + M_(1, 2)*M_(1, 2);
	if (fabs(p1) < 0.00001){
		// A is diagonal.
		(*pE_)(0) = M_(0, 0);
		(*pE_)(1) = M_(1, 1);
		(*pE_)(2) = M_(2, 2);
	}
	else{
		T q = M_(0, 0) + M_(1, 1) + M_(2, 2); q /= 3;
		T t1 = M_(0, 0) - q;
		T t2 = M_(1, 1) - q;
		T t3 = M_(2, 2) - q;
		T p2 = t1*t1 + t2*t2 + t3*t3 + 2 * p1;
		T p = sqrt(p2 / 6);
		Matrix<T, 3, 3> B = (1 / p)*(M_ - q*Matrix<T, 3, 3>::Identity());
		T r = B.determinant() / 2;

		// In exact arithmetic for a symmetric matrix - 1 <= r <= 1
		// but computation error can leave it slightly outside this range.
		T phi;
		if (r <= -1)
			phi = T(3.141592653589793238) / 3;
		else if (r >= 1)
			phi = 0;
		else
			phi = acos(r) / 3;

		// the eigenvalues satisfy eig3 <= eig2 <= eig1
		(*pE_)(0) = q + 2 * p * cos(phi);
		(*pE_)(2) = q + 2 * p * cos(phi + (2 * T(3.141592653589793238) / 3));
		(*pE_)(1) = 3 * q - (*pE_)(0) - (*pE_)(2);  //since trace(A) = eig1 + eig2 + eig3
	}
	//solve out eigen vectors
	if (pV_){
		Matrix<T, 3, 3> s0 = ((*pE_)(0)*Matrix<T, 3, 3>::Identity() - M_);
		Matrix<T, 3, 3> s1 = ((*pE_)(1)*Matrix<T, 3, 3>::Identity() - M_);
		Matrix<T, 3, 3> s2 = ((*pE_)(2)*Matrix<T, 3, 3>::Identity() - M_);
		{
			Matrix<T, 3, 3> s = s0 * s1;
			(*pV_).col(2) = s.col(2).normalized();
		}
		{
			Matrix<T, 3, 3> s = s0 * s2;
			(*pV_).col(1) = s.col(1).normalized();
		}
		{
			Matrix<T, 3, 3> s = s1 * s2;
			(*pV_).col(0) = s.col(0).normalized();
		}
	}
	return;
}

#endif
