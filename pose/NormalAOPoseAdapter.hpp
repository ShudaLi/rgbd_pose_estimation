#ifndef _NORMAL_AO_POSE_ADAPTER_HEADER_
#define _NORMAL_AO_POSE_ADAPTER_HEADER_

#include <stdlib.h>
#include <vector>
#include <AOPoseAdapter.hpp>
#include <limits>

using namespace std;
using namespace Eigen;
/**
 * \brief The namespace for the absolute pose methods.
 */


template<typename POSE_T, typename POINT_T>
class NormalAOPoseAdapter : public AOPoseAdapter<POSE_T, POINT_T>
{

protected:
	using PoseAdapterBase<POSE_T, POINT_T>::_t_w;
	using PoseAdapterBase<POSE_T, POINT_T>::_R_cw;
	using PnPPoseAdapter<POSE_T, POINT_T>::_bearingVectors;
	using PnPPoseAdapter<POSE_T, POINT_T>::_points_g;
	using AOPoseAdapter<POSE_T, POINT_T>::_points_c;

public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	typedef typename PoseAdapterBase<POSE_T, POINT_T>::Vector3 Vector3;
	typedef typename PoseAdapterBase<POSE_T, POINT_T>::SO3_T SO3_T;
	typedef typename PnPPoseAdapter<POSE_T, POINT_T>::MatrixX MatrixX;

  /**
   * \brief Constructor. See protected class-members to understand parameters
   */
   NormalAOPoseAdapter(
      const MatrixX & bearingVectors,
	  const MatrixX & points_c,
	  const MatrixX & normal_c,
	  const MatrixX & points_g,
	  const MatrixX & normal_g);
  /**
   * \brief Constructor. See protected class-members to understand parameters
   */
   NormalAOPoseAdapter(
	  const MatrixX & bearingVectors,
	  const MatrixX & points_c,
	  const MatrixX & normal_c,
	  const MatrixX & points_g,
	  const MatrixX & normal_g,
	  const SO3_T & R);
  /**
   * \brief Constructor. See protected class-members to understand parameters
   */
  NormalAOPoseAdapter(
	  const MatrixX & bearingVectors,
	  const MatrixX & points_c,
	  const MatrixX & normal_c,
	  const MatrixX & points_g,
	  const MatrixX & normal_g,
	  const Vector3 & t,
      const SO3_T & R );
  /**
   * Destructor
   */
  virtual ~NormalAOPoseAdapter();

  //Access of correspondences
  bool isInlierNN(int index) const;
  POINT_T weightNN(int index) const;
  /** See parent-class */
  virtual Point3 getNormalCurr( int index ) const;
  virtual Point3 getNormalGlob( int index ) const;
  virtual void setInlier(const Matrix<short, Dynamic, Dynamic>& inliers);
  virtual void setWeights(const Matrix<short, Dynamic, Dynamic>& weights);
  virtual void printInlier() const;
  const vector<short>& getInlierIdx() const { return _vInliersNN; }
  void cvtInlier();

protected:
  /** Reference to the 3-D points in the camera-frame */
	const MatrixX & _normal_c;
	const MatrixX & _normal_g;

	/** flags of inliers. */
	Matrix<short, Dynamic, 1> _inliers_nl;
	Matrix<short, Dynamic, 1> _weights_nl;

	vector<short> _vInliersNN;
};


template<typename POSE_T, typename POINT_T>
NormalAOPoseAdapter<POSE_T, POINT_T>::NormalAOPoseAdapter(
	const MatrixX & bearingVectors,
	const MatrixX & points_c,
	const MatrixX & normal_c,
	const MatrixX & points_g,
	const MatrixX & normal_g ) :
	AOPoseAdapter<POSE_T, POINT_T>(bearingVectors, points_c, points_g),
	_normal_c(normal_c),
	_normal_g(normal_g)
{
	_inliers_nl.resize(_bearingVectors.cols());
	_inliers_nl.setOnes();
}

template<typename POSE_T, typename POINT_T>
NormalAOPoseAdapter<POSE_T, POINT_T>::NormalAOPoseAdapter(
	const MatrixX & bearingVectors,
	const MatrixX & points_c,
	const MatrixX & normal_c,
	const MatrixX & points_g,
	const MatrixX & normal_g,
	const SO3_T & R) :
	AOPoseAdapter<POSE_T, POINT_T>(bearingVectors, points_c, points_g, R),
	_normal_c(normal_c),
	_normal_g(normal_g)
{
	_inliers_nl.resize(_bearingVectors.cols());
	_inliers_nl.setOnes();
}

template<typename POSE_T, typename POINT_T>
NormalAOPoseAdapter<POSE_T, POINT_T>::NormalAOPoseAdapter(
	const MatrixX & bearingVectors,
	const MatrixX & points_c,
	const MatrixX & normal_c,
	const MatrixX & points_g,
	const MatrixX & normal_g, 
	const Vector3 & t,
	const SO3_T & R) :
	AOPoseAdapter<POSE_T, POINT_T>(bearingVectors, points_c, points_g, t, R),
	_normal_c(normal_c),
	_normal_g(normal_g)
{
	_inliers_nl.resize(_bearingVectors.cols());
	_inliers_nl.setOnes();
}

template<typename POSE_T, typename POINT_T>
NormalAOPoseAdapter<POSE_T, POINT_T>::~NormalAOPoseAdapter()
{}

template<typename POSE_T, typename POINT_T>
bool NormalAOPoseAdapter<POSE_T, POINT_T>::isInlierNN(int index) const
{
	assert(index < _inliers_nl.rows());
	return _inliers_nl(index) == 1;
}

template<typename POSE_T, typename POINT_T>
POINT_T NormalAOPoseAdapter<POSE_T, POINT_T>::weightNN(int index) const
{
	if (!_weights_nl.rows()) return POINT_T(1.0);
	else{
		assert(index < _weights_nl.rows());
		return POINT_T(_weights_nl(index)) / numeric_limits<short>::max();
	}
}

template<typename POSE_T, typename POINT_T>
typename NormalAOPoseAdapter<POSE_T, POINT_T>::Point3 NormalAOPoseAdapter<POSE_T, POINT_T>::getNormalCurr(
	int index) const
{
	assert(index < _bearingVectors.cols() );
	return _normal_c.col(index);
}

template<typename POSE_T, typename POINT_T>
typename NormalAOPoseAdapter<POSE_T, POINT_T>::Point3 NormalAOPoseAdapter<POSE_T, POINT_T>::getNormalGlob(
	int index) const
{
	assert(index < _bearingVectors.cols());
	return _normal_g.col(index);
}

template <typename POSE_T, typename POINT_T>
void NormalAOPoseAdapter<POSE_T, POINT_T>::setInlier(const Matrix<short, Dynamic, Dynamic>& inliers)
{
	assert(inliers.rows() == _inliers_nl.rows());
	if (inliers.cols() == 1){
		PnPPoseAdapter<POSE_T, POINT_T>::setInlier(inliers);
	}
	if (inliers.cols() == 2){
		AOPoseAdapter<POSE_T, POINT_T>::setInlier(inliers);
	}
	if (inliers.cols() == 3){
		AOPoseAdapter<POSE_T, POINT_T>::setInlier(inliers);
		//_inliers_nl = inliers.col(2);
		memcpy(_inliers_nl.data(), inliers.col(2).data(), inliers.rows() * 2);
	}
	return;
}

template <typename POSE_T, typename POINT_T>
void NormalAOPoseAdapter<POSE_T, POINT_T>::setWeights(const Matrix<short, Dynamic, Dynamic>& weights)
{
	if (weights.cols() == 1){
		PnPPoseAdapter<POSE_T, POINT_T>::setWeights(weights);
	}
	if (weights.cols() == 2){
		AOPoseAdapter<POSE_T, POINT_T>::setWeights(weights);
	}
	if (weights.cols() == 3){
		AOPoseAdapter<POSE_T, POINT_T>::setWeights(weights);
		_weights_nl = weights.col(2);
		//memcpy(_weights_nl.data(), weights.col(2).data(), weights.rows() * 2);
	}
	return;
}

template <typename POSE_T, typename POINT_T>
void NormalAOPoseAdapter<POSE_T, POINT_T>::printInlier() const
{
	AOPoseAdapter<POSE_T, POINT_T>::printInlier();
	cout << _inliers_nl.transpose() << endl;
}

template <typename POSE_T, typename POINT_T>
void NormalAOPoseAdapter<POSE_T, POINT_T>::cvtInlier()
{
	_vInliersNN.clear();
	_vInliersNN.reserve(getNumberCorrespondences());
	for (short r = 0; r < (short)_inliers_nl.rows(); r++) {
		if (1 == _inliers_nl[r]){
			_vInliersNN.push_back(r);
		}
	}
}

#endif 
