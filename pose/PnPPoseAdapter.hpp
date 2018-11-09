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
#include <Utility.hpp>

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
class PnPPoseAdapter : public PoseAdapterBase<Tp>
{
protected:
	using PoseAdapterBase<Tp>::_t_w;
	using PoseAdapterBase<Tp>::_R_cw;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef typename PoseAdapterBase<Tp>::Vector3 Vector3;
  typedef typename PoseAdapterBase<Tp>::SO3_T SO3_T;
  typedef typename PoseAdapterBase<Tp>::Point3 Point3;

  typedef Matrix<Tp, Dynamic, Dynamic> MatrixX;

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
  virtual Tp getWeight( int index ) const;
  /** See parent-class */
  virtual Point3 getPointGlob( int index ) const;
  /** See parent-class */
  virtual int getNumberCorrespondences() const;

  virtual void setInlier(const Matrix<short, Dynamic, Dynamic>& inliers);
  virtual void setWeights(const Matrix<Tp, Dynamic, Dynamic>& weights);
  virtual void printInlier() const;
  const vector<short>& getInlierIdx() const { return _vInliersPnP;}
  void cvtInlier();

  Tp getError(int index) const;

  void setMaxVotes(int votes){  _max_votes = votes;  };

  int getMaxVotes() { return _max_votes; };

  bool isInlier23(int index) const;
  Tp weight23(int index) const;
  void sortIdx();
  void getSortedIdx(vector<int>& select_) const;
protected:
	/** Reference to the bearing-vectors expressed in the camera-frame */
	const MatrixX & _bearingVectors; //normalized 2d homogeneous coordinate
	/** Reference to the points expressed in the world/global-frame. */
	const MatrixX & _points_g;
	/** flags of inliers. */
	Matrix<short, Dynamic, 1> _inliers;
	Matrix<Tp, Dynamic, 1> _weights;
    vector<int> _idx; //stores the sorted index 
	vector<short> _vInliersPnP;
	/** max votes. */
	int _max_votes;
};


template <typename Tp>
PnPPoseAdapter<Tp>::PnPPoseAdapter(
	const MatrixX & bearingVectors,
	const MatrixX & points) :
	PoseAdapterBase<Tp>(),
	_bearingVectors(bearingVectors),
	_points_g(points)
{
	_inliers.resize(_bearingVectors.cols());
	_inliers.setOnes();
}

template <typename Tp>
PnPPoseAdapter<Tp>::PnPPoseAdapter(
	const MatrixX & bearingVectors,
	const MatrixX & points,
	const SO3_T & R) :
	PoseAdapterBase<Tp>(R),
	_bearingVectors(bearingVectors),
	_points_g(points)
{
	_inliers.resize(_bearingVectors.cols());
	_inliers.setOnes();
}

template <typename Tp>
PnPPoseAdapter<Tp>::PnPPoseAdapter(
	const MatrixX & bearingVectors,
	const MatrixX & points,
	const Vector3 & t,
	const SO3_T & R) :
	PoseAdapterBase<Tp>(t, R),
	_bearingVectors(bearingVectors),
	_points_g(points)
{
	_inliers.resize(_bearingVectors.cols());
	_inliers.setOnes();
}

template <typename Tp>
typename PnPPoseAdapter<Tp>::Point3 PnPPoseAdapter<Tp>::
getBearingVector(int index) const
{
	assert(index < _bearingVectors.cols());
	return _bearingVectors.col(index);
}


template <typename Tp>
Tp PnPPoseAdapter<Tp>::
getWeight(int index) const
{
	return Tp(1.);
}

template <typename Tp>
typename PnPPoseAdapter<Tp>::Point3 PnPPoseAdapter<Tp>::
getPointGlob(int index) const
{
	assert(index < _bearingVectors.cols());
	return _points_g.col(index);
}

template<typename Tp>
bool PnPPoseAdapter<Tp>::isInlier23(int index) const
{
	assert(index < _inliers.rows());
	return _inliers(index) == 1;
}

template<typename Tp>
Tp PnPPoseAdapter<Tp>::weight23(int index) const
{
	if (!_weights.rows()) return Tp(1.0);
	else{
		assert(index < _weights.rows());
		return _weights(index);
	}
}

template <typename Tp>
int PnPPoseAdapter<Tp>::getNumberCorrespondences() const
{
	return _bearingVectors.cols();
}

template <typename Tp>
void PnPPoseAdapter<Tp>::setInlier(const Matrix<short, Dynamic, Dynamic>& inliers)
{
	assert(inliers.rows() == _inliers.rows());
	//_inliers = inliers.col(0);
	memcpy(_inliers.data(), inliers.data(), inliers.rows()*2);
}

template <typename Tp>
Tp PnPPoseAdapter<Tp>::getError(int index) const
{
	Point3 Xc = _R_cw * getPointGlob(index) + _t_w;
	Xc.normalize();
	return Xc.cross(getBearingVector(index)).norm();
}

template <typename Tp>
void PnPPoseAdapter<Tp>::setWeights(const Matrix<Tp, Dynamic, Dynamic>& weights)
{
	_weights = weights.col(0);
	//memcpy(_weights.data(), weights.data(), weights.rows() * 2);

	return;
}

template <typename Tp>
void PnPPoseAdapter<Tp>::printInlier() const
{
	cout << _inliers.transpose() << endl;
}

template <typename Tp>
void PnPPoseAdapter<Tp>::cvtInlier()
{
	_vInliersPnP.clear();
	_vInliersPnP.reserve(getNumberCorrespondences());
	for (short r = 0; r < (short)_inliers.rows(); r++) {
		if (1 == _inliers[r]){
			_vInliersPnP.push_back(r);
		}
	}
}

template <typename Tp>
void PnPPoseAdapter<Tp>::sortIdx(){
    //sort the index according to weights 
    vector<Tp> weigh(_weights.data(), _weights.data() + _weights.rows() * _weights.cols());
    _idx = sortIndexes<Tp>( weigh );
}

template <typename Tp>
void PnPPoseAdapter<Tp>::getSortedIdx(vector<int>& select_) const{
    //sort the index according to weights 
    for (int i = 0; i < (int)select_.size(); ++i)
    {
    	int j = select_[i];
    	if(j < (int)_idx.size())
    		select_[i] = _idx[j];
    }
}

#endif 
