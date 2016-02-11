/**
 * \file CentralAbsoluteAdapter.hpp
 * \brief Adapter-class for passing bearing-vector-to-point correspondences to
 *        the central absolute-pose algorithms. It maps opengv types
 *        back to opengv types.
 */

#ifndef _PNP_POSE_ADAPTER_HEADER_
#define _PNP_POSE_ADAPTER_HEADER_

#include <stdlib.h>
#include <vector>
#include <PoseAdapterBase.hpp>

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
template<typename POSE_T, typename POINT_T>
class PnPPoseAdapter : public PoseAdapterBase<POSE_T, POINT_T>
{
protected:
	using PoseAdapterBase<POSE_T, POINT_T>::_t_w;
	using PoseAdapterBase<POSE_T, POINT_T>::_R_cw;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef typename PoseAdapterBase<POSE_T, POINT_T>::Vector3 Vector3;
  typedef typename PoseAdapterBase<POSE_T, POINT_T>::SO3_T SO3_T;
  typedef typename PoseAdapterBase<POSE_T, POINT_T>::Point3 Point3;

  typedef Matrix<POINT_T, Dynamic, Dynamic> MatrixX;

  /**
   * \brief Constructor. See protected class-members to understand parameters
   */
  PnPPoseAdapter(
      const MatrixX & bearingVectors,
	  const MatrixX & points );
  /**
   * \brief Constructor. See protected class-members to understand parameters
   */
  PnPPoseAdapter(
	  const MatrixX & bearingVectors,
	  const MatrixX & points,
      const SO3_T & R );
  /**
   * \brief Constructor. See protected class-members to understand parameters
   */
  PnPPoseAdapter(
	  const MatrixX & bearingVectors,
	  const MatrixX & points,
      const Vector3 & t,
      const SO3_T & R );
  /**
   * Destructor
   */
  virtual ~PnPPoseAdapter(){};

  //Access of correspondences
  
  /** See parent-class */
  virtual Point3 getBearingVector( int index ) const;
  /** See parent-class */
  virtual POINT_T getWeight( int index ) const;
  /** See parent-class */
  virtual Point3 getPointGlob( int index ) const;
  /** See parent-class */
  virtual int getNumberCorrespondences() const;

  virtual void setInlier(const Matrix<short, Dynamic, Dynamic>& inliers);
  virtual void setWeights(const Matrix<short, Dynamic, Dynamic>& weights);
  virtual void printInlier() const;
  const vector<short>& getInlierIdx() const { return _vInliersPnP;}
  void cvtInlier();

  POINT_T getError(int index) const;

  void setMaxVotes(int votes){  _max_votes = votes;  };

  int getMaxVotes() { return _max_votes; };

  bool isInlier23(int index) const;
  POINT_T weight23(int index) const;
protected:
	/** Reference to the bearing-vectors expressed in the camera-frame */
	const MatrixX & _bearingVectors; //normalized 2d homogeneous coordinate
	/** Reference to the points expressed in the world/global-frame. */
	const MatrixX & _points_g;
	/** flags of inliers. */
	Matrix<short, Dynamic, 1> _inliers;
	Matrix<short, Dynamic, 1> _weights;
	vector<short> _vInliersPnP;
	/** max votes. */
	int _max_votes;
};


template <typename POSE_T, typename POINT_T>
PnPPoseAdapter<POSE_T, POINT_T>::PnPPoseAdapter(
	const MatrixX & bearingVectors,
	const MatrixX & points) :
	PoseAdapterBase<POSE_T, POINT_T>(),
	_bearingVectors(bearingVectors),
	_points_g(points)
{
	_inliers.resize(_bearingVectors.cols());
	_inliers.setOnes();
}

template <typename POSE_T, typename POINT_T>
PnPPoseAdapter<POSE_T, POINT_T>::PnPPoseAdapter(
	const MatrixX & bearingVectors,
	const MatrixX & points,
	const SO3_T & R) :
	PoseAdapterBase<POSE_T, POINT_T>(R),
	_bearingVectors(bearingVectors),
	_points_g(points)
{
	_inliers.resize(_bearingVectors.cols());
	_inliers.setOnes();
}

template <typename POSE_T, typename POINT_T>
PnPPoseAdapter<POSE_T, POINT_T>::PnPPoseAdapter(
	const MatrixX & bearingVectors,
	const MatrixX & points,
	const Vector3 & t,
	const SO3_T & R) :
	PoseAdapterBase<POSE_T, POINT_T>(t, R),
	_bearingVectors(bearingVectors),
	_points_g(points)
{
	_inliers.resize(_bearingVectors.cols());
	_inliers.setOnes();
}

template <typename POSE_T, typename POINT_T>
typename PnPPoseAdapter<POSE_T, POINT_T>::Point3 PnPPoseAdapter<POSE_T, POINT_T>::
getBearingVector(int index) const
{
	assert(index < _bearingVectors.cols());
	return _bearingVectors.col(index);
}


template <typename POSE_T, typename POINT_T>
POINT_T PnPPoseAdapter<POSE_T, POINT_T>::
getWeight(int index) const
{
	return POINT_T(1.);
}

template <typename POSE_T, typename POINT_T>
typename PnPPoseAdapter<POSE_T, POINT_T>::Point3 PnPPoseAdapter<POSE_T, POINT_T>::
getPointGlob(int index) const
{
	assert(index < _bearingVectors.cols());
	return _points_g.col(index);
}

template<typename POSE_T, typename POINT_T>
bool PnPPoseAdapter<POSE_T, POINT_T>::isInlier23(int index) const
{
	assert(index < _inliers.rows());
	return _inliers(index) == 1;
}

template<typename POSE_T, typename POINT_T>
POINT_T PnPPoseAdapter<POSE_T, POINT_T>::weight23(int index) const
{
	if (!_weights.rows()) return POINT_T(1.0);
	else{
		assert(index < _weights.rows());
		return POINT_T(_weights(index)) / numeric_limits<short>::max();
	}
}

template <typename POSE_T, typename POINT_T>
int PnPPoseAdapter<POSE_T, POINT_T>::getNumberCorrespondences() const
{
	return _bearingVectors.cols();
}

template <typename POSE_T, typename POINT_T>
void PnPPoseAdapter<POSE_T, POINT_T>::setInlier(const Matrix<short, Dynamic, Dynamic>& inliers)
{
	assert(inliers.rows() == _inliers.rows());
	//_inliers = inliers.col(0);
	memcpy(_inliers.data(), inliers.data(), inliers.rows()*2);
}

template <typename POSE_T, typename POINT_T>
POINT_T PnPPoseAdapter<POSE_T, POINT_T>::getError(int index) const
{
	Point3 Xc = _Rcw * getPointGlob(index) + _t;
	Xc.normalize();
	return Xc.cross(getBearingVector(index)).norm();
}

template <typename POSE_T, typename POINT_T>
void PnPPoseAdapter<POSE_T, POINT_T>::setWeights(const Matrix<short, Dynamic, Dynamic>& weights)
{
	_weights = weights.col(0);
	//memcpy(_weights.data(), weights.data(), weights.rows() * 2);

	return;
}

template <typename POSE_T, typename POINT_T>
void PnPPoseAdapter<POSE_T, POINT_T>::printInlier() const
{
	cout << _inliers.transpose() << endl;
}

template <typename POSE_T, typename POINT_T>
void PnPPoseAdapter<POSE_T, POINT_T>::cvtInlier()
{
	_vInliersPnP.clear();
	_vInliersPnP.reserve(getNumberCorrespondences());
	for (short r = 0; r < (short)_inliers.rows(); r++) {
		if (1 == _inliers[r]){
			_vInliersPnP.push_back(r);
		}
	}
}


#endif 
