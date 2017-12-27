/**
 * \file CentralAbsoluteAdapter.hpp
 * \brief Adapter-class for passing bearing-vector-to-point correspondences to
 *        the central absolute-pose algorithms. It maps opengv types
 *        back to opengv types.
 */

#ifndef _AO_POSE_ADAPTER_HEADER_
#define _AO_POSE_ADAPTER_HEADER_

#include <stdlib.h>
#include <vector>
#include <PnPPoseAdapter.hpp>

using namespace std;
using namespace Eigen;
/**
 * \brief The namespace for the absolute pose methods.
 */

/**
 * Check the documentation of the parent-class to understand the meaning of
 * an AbsoluteAdapter. This child-class is for the central case and holds data
 * in form of references to opengv-types.
 */
template<typename Tp>
class AOPoseAdapter : public PnPPoseAdapter<Tp>
{

protected:
	using PoseAdapterBase<Tp>::_t_w;
	using PoseAdapterBase<Tp>::_R_cw;
	using PnPPoseAdapter<Tp>::_bearingVectors;
	using PnPPoseAdapter<Tp>::_points_g;

public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	typedef typename PoseAdapterBase<Tp>::Vector3 Vector3;
	typedef typename PoseAdapterBase<Tp>::SO3_T SO3_T;
	typedef typename PoseAdapterBase<Tp>::Point3 Point3;
	
	typedef typename PnPPoseAdapter<Tp>::MatrixX MatrixX;

  /**
   * \brief Constructor. See protected class-members to understand parameters
   */
   AOPoseAdapter(
      const MatrixX & bearingVectors,
	  const MatrixX & points_c,
	  const MatrixX & points_g);
  /**
   * \brief Constructor. See protected class-members to understand parameters
   */
  AOPoseAdapter(
	  const MatrixX & bearingVectors,
	  const MatrixX & points_c,
	  const MatrixX & points_g,
      const SO3_T & R );
  /**
   * \brief Constructor. See protected class-members to understand parameters
   */
  AOPoseAdapter(
	  const MatrixX & bearingVectors,
	  const MatrixX & points_c,
	  const MatrixX & points_g,
      const Vector3 & t,
      const SO3_T & R );
  /**
   * Destructor
   */
  virtual ~AOPoseAdapter();

  //Access of correspondences
  
  bool isInlier33(int index) const;
  Tp weight33(int index) const;
  /** See parent-class */
  virtual Point3 getPointCurr( int index ) const;
  virtual bool isValid( int index ) const;
  virtual void setInlier(const Matrix<short, Dynamic, Dynamic>& inliers);
  virtual void setWeights(const Matrix<short, Dynamic, Dynamic>& weights);
  virtual void printInlier() const;
  const vector<short>& getInlierIdx() const { return _vInliersAO; }
  void cvtInlier();

protected:
  /** Reference to the 3-D points in the camera-frame */
  const MatrixX & _points_c; 
  /** flags of inliers. */
  Matrix<short, Dynamic, 1> _inliers_3d;
  Matrix<short, Dynamic, 1> _weights_3d;
  vector<short> _vInliersAO;
};


template<typename Tp>
AOPoseAdapter<Tp>::AOPoseAdapter(
	const MatrixX & bearingVectors,
	const MatrixX & points_c,
	const MatrixX & points_g) :
	PnPPoseAdapter<Tp>(bearingVectors, points_g),
	_points_c(points_c)
{
	_inliers_3d.resize(_bearingVectors.cols());
	_inliers_3d.setOnes();
}

template<typename Tp>
AOPoseAdapter<Tp>::AOPoseAdapter(
	const MatrixX & bearingVectors,
	const MatrixX & points_c,
	const MatrixX & points_g,
	const SO3_T & R) :
	PnPPoseAdapter<Tp>(bearingVectors, points_g, R),
	_points_c(points_c)
{
	_inliers_3d.resize(_bearingVectors.cols());
	_inliers_3d.setOnes();
}

template<typename Tp>
AOPoseAdapter<Tp>::AOPoseAdapter(
	const MatrixX & bearingVectors,
	const MatrixX & points_c,
	const MatrixX & points_g,
	const Vector3 & t,
	const SO3_T & R) :
	PnPPoseAdapter<Tp>(bearingVectors, points_g, t, R),
	_points_c(points_c)
{
	_inliers_3d.resize(_bearingVectors.cols());
	_inliers_3d.setOnes();
}

template<typename Tp>
AOPoseAdapter<Tp>::~AOPoseAdapter()
{}

template<typename Tp>
typename AOPoseAdapter<Tp>::Point3 AOPoseAdapter<Tp>::getPointCurr(
	int index) const
{
	assert(index < _bearingVectors.cols() );
	return _points_c.col(index);
}

template<typename Tp>
bool AOPoseAdapter<Tp>::isValid( int index ) const
{
	assert(index < _bearingVectors.cols());
	return _points_c.col(index)(0) == _points_c.col(index)(0) || _points_c.col(index)(1) == _points_c.col(index)(1) || _points_c.col(index)(2) == _points_c.col(index)(2);
}

template<typename Tp>
bool AOPoseAdapter<Tp>::isInlier33(int index) const
{
	assert(index < _inliers_3d.rows());
	return _inliers_3d(index) == 1;
}

template<typename Tp>
Tp AOPoseAdapter<Tp>::weight33(int index) const
{
	if (!_weights_3d.rows()) return Tp(1.0);
	else{
		assert(index < _weights_3d.rows());
		return Tp(_weights_3d(index)) / numeric_limits<short>::max();
	}
}

template <typename Tp>
void AOPoseAdapter<Tp>::setInlier(const Matrix<short, Dynamic, Dynamic>& inliers)
{
	assert(inliers.rows() == _inliers_3d.rows());
	if (inliers.cols() == 1){
		PnPPoseAdapter<Tp>::setInlier(inliers);
	}
	else{
		PnPPoseAdapter<Tp>::setInlier(inliers);
		_inliers_3d = inliers.col(1);
		//memcpy(_inliers_3d.data(), inliers.col(1).data(), inliers.rows() * 2);
	}
	return;
}

template <typename Tp>
void AOPoseAdapter<Tp>::setWeights(const Matrix<short, Dynamic, Dynamic>& weights)
{
	if (weights.rows() == 1){
		PnPPoseAdapter<Tp>::setWeights(weights);
	}
	else{
		PnPPoseAdapter<Tp>::setWeights(weights);
		_weights_3d = weights.col(1);
		//memcpy(_weights_3d.data(), weights.col(1).data(), weights.rows() * 2);
	}
	return;
}

template <typename Tp>
void AOPoseAdapter<Tp>::printInlier() const
{
	PnPPoseAdapter<Tp>::printInlier();
	cout << _inliers_3d.transpose() << endl;
}

template <typename Tp>
void AOPoseAdapter<Tp>::cvtInlier()
{
	_vInliersAO.clear();
	_vInliersAO.reserve(getNumberCorrespondences());
	for (short r = 0; r < (short)_inliers_3d.rows(); r++) {
		if (1 == _inliers_3d[r]){
			_vInliersAO.push_back(r);
		}
	}
}

#endif 
