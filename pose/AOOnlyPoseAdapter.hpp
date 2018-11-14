/**
 * \file CentralAbsoluteAdapter.hpp
 * \brief Adapter-class for passing bearing-vector-to-point correspondences to
 *        the central absolute-pose algorithms. It maps opengv types
 *        back to opengv types.
 */

#ifndef _AO_ONLY_POSE_ADAPTER_HEADER_
#define _AO_ONLY_POSE_ADAPTER_HEADER_

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
template<typename Tp>
class AOOnlyPoseAdapter : public PoseAdapterBase<Tp>
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
   AOOnlyPoseAdapter(
	  const MatrixX & points_c,
	  const MatrixX & points_g);
  /**
   * \brief Constructor. See protected class-members to understand parameters
   */
  AOOnlyPoseAdapter(
	  const MatrixX & points_c,
	  const MatrixX & points_g,
      const SO3_T & R );
  /**
   * \brief Constructor. See protected class-members to understand parameters
   */
  AOOnlyPoseAdapter(
	  const MatrixX & points_c,
	  const MatrixX & points_g,
      const Vector3 & t,
      const SO3_T & R );
  /**
   * Destructor
   */
  virtual ~AOOnlyPoseAdapter();

  //Access of correspondences
  
  bool isInlier33(int index) const;
  Tp weight33(int index) const;
  /** See parent-class */
  virtual Point3 getBearingVector(int index) const { return Point3();} 
  virtual Point3 getPointCurr( int index ) const;
  virtual Point3 getPointGlob(int index) const;
  virtual Tp getWeight(int index) const{ return Tp(1.); }
  virtual int getNumberCorrespondences() const { return _points_g.cols(); }

  void setMaxVotes(int votes){  _max_votes = votes;  };

  int getMaxVotes() { return _max_votes; };


  virtual bool isValid( int index ) const;
  virtual void setInlier(const Matrix<short, Dynamic, Dynamic>& inliers);
  virtual void setWeights(const Matrix<Tp, Dynamic, Dynamic>& weights);
  virtual void printInlier() const;
  const vector<short>& getInlierIdx() const { return _vInliersAO; }
  void cvtInlier();
  void sortIdx();
  void getSortedIdx(vector<int>& select_) const;
protected:
  /** Reference to the 3-D points in the camera-frame */
  const MatrixX & _points_c; 
  const MatrixX & _points_g; 
  /** flags of inliers. */
  Matrix<short, Dynamic, 1> _inliers_3d;
  Matrix<Tp, Dynamic, 1> _weights_3d;
  vector<int> _idx; //stores the sorted index 
  vector<short> _vInliersAO;
  int _max_votes;
};


template<typename Tp>
AOOnlyPoseAdapter<Tp>::AOOnlyPoseAdapter(
	const MatrixX & points_c,
	const MatrixX & points_g) :
	PoseAdapterBase<Tp>(),
	_points_c(points_c), _points_g(points_g)
{
	_inliers_3d.resize(points_c.cols());
	_inliers_3d.setOnes();
}

template<typename Tp>
AOOnlyPoseAdapter<Tp>::AOOnlyPoseAdapter(
	const MatrixX & points_c,
	const MatrixX & points_g,
	const SO3_T & R) :
	PoseAdapterBase<Tp>(R),
	_points_c(points_c), _points_g(points_g)
{
	_inliers_3d.resize(points_c.cols());
	_inliers_3d.setOnes();
}

template<typename Tp>
AOOnlyPoseAdapter<Tp>::AOOnlyPoseAdapter(
	const MatrixX & points_c,
	const MatrixX & points_g,
	const Vector3 & t,
	const SO3_T & R) :
	PoseAdapterBase<Tp>(t, R),
	_points_c(points_c), _points_g(points_g)
{
	_inliers_3d.resize(points_c.cols());
	_inliers_3d.setOnes();
}

template<typename Tp>
AOOnlyPoseAdapter<Tp>::~AOOnlyPoseAdapter()
{}

template<typename Tp>
typename AOOnlyPoseAdapter<Tp>::Point3 AOOnlyPoseAdapter<Tp>::getPointCurr(
	int index) const
{
	assert(index < _points_c.cols() );
	return _points_c.col(index);
}

template <typename Tp>
typename AOOnlyPoseAdapter<Tp>::Point3 AOOnlyPoseAdapter<Tp>::getPointGlob(
	int index) const
{
	assert(index < _points_c.cols());
	return _points_g.col(index);
}

template<typename Tp>
bool AOOnlyPoseAdapter<Tp>::isValid( int index ) const
{
	assert(index < _points_c.cols());
	return _points_c.col(index)(0) == _points_c.col(index)(0) || _points_c.col(index)(1) == _points_c.col(index)(1) || _points_c.col(index)(2) == _points_c.col(index)(2);
}

template<typename Tp>
bool AOOnlyPoseAdapter<Tp>::isInlier33(int index) const
{
	assert(index < _inliers_3d.rows());
	return _inliers_3d(index) == 1;
}

template<typename Tp>
Tp AOOnlyPoseAdapter<Tp>::weight33(int index) const
{
	if (!_weights_3d.rows()) return Tp(1.0);
	else{
		assert(index < _weights_3d.rows());
		return _weights_3d(index);
	}
}

template <typename Tp>
void AOOnlyPoseAdapter<Tp>::setInlier(const Matrix<short, Dynamic, Dynamic>& inliers)
{
	assert(inliers.rows() == _inliers_3d.rows());
	if (inliers.cols() == 1){
		// PnPPoseAdapter<Tp>::setInlier(inliers);
	}
	else{
		// PnPPoseAdapter<Tp>::setInlier(inliers);
		_inliers_3d = inliers.col(1);
		//memcpy(_inliers_3d.data(), inliers.col(1).data(), inliers.rows() * 2);
	}
	return;
}

template <typename Tp>
void AOOnlyPoseAdapter<Tp>::setWeights(const Matrix<Tp, Dynamic, Dynamic>& weights)
{
	if (weights.rows() == 1){
		// PnPPoseAdapter<Tp>::setWeights(weights);
	}
	else{
		// PnPPoseAdapter<Tp>::setWeights(weights);
		_weights_3d = weights.col(1);
		//memcpy(_weights_3d.data(), weights.col(1).data(), weights.rows() * 2);
	}
	return;
}

template <typename Tp>
void AOOnlyPoseAdapter<Tp>::printInlier() const
{
	// PnPPoseAdapter<Tp>::printInlier();
	cout << _inliers_3d.transpose() << endl;
}

template <typename Tp>
void AOOnlyPoseAdapter<Tp>::cvtInlier()
{
	_vInliersAO.clear();
	_vInliersAO.reserve(this->getNumberCorrespondences());
	for (short r = 0; r < (short)_inliers_3d.rows(); r++) {
		if (1 == _inliers_3d[r]){
			_vInliersAO.push_back(r);
		}
	}
}

template <typename Tp>
void AOOnlyPoseAdapter<Tp>::sortIdx(){
    //sort the index according to weights 
    vector<Tp> weigh(_weights_3d.data(), _weights_3d.data() + _weights_3d.rows() * _weights_3d.cols());
    _idx = sortIndexes<Tp>( weigh );
    // for (int i = 0; i < _idx.size(); ++i)
    // {
    //     cout << _weights_3d.data()[_idx[i]] << " ";
    // }
    // cout << endl;
}

template <typename Tp>
void AOOnlyPoseAdapter<Tp>::getSortedIdx(vector<int>& select_) const{
    //sort the index according to weights 
    for (int i = 0; i < (int)select_.size(); ++i)
    {
        int j = select_[i];
        if(j < (int)_idx.size())
            select_[i] = _idx[j];
    }
}


#endif 
