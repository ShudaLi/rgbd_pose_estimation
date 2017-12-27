/**
 * \file based on Kneip's opengv AbsoluteAdapterBase.hpp
 * \brief Adapter-class for passing bearing-vector-to-point correspondences to
 *        the absolute-pose algorithms.
 */

#ifndef _POSE_ADAPTERBASE_HEADER_
#define _POSE_ADAPTERBASE_HEADER_

#include <stdlib.h>
#include <vector>
#include <Eigen/Eigen>
#include <Eigen/src/Core/util/DisableStupidWarnings.h>
#include <se3.hpp>

using namespace Eigen;
/**
 * \brief The namespace for the absolute pose methods.
 */

/**
 * The PoseAdapterBase is the base-class for the visitors to absolute pose 
 * algorithms. It provides a unified interface to opengv-methods to access 
 * bearing-vectors, world points, priors or known variables for the absolute 
 * pose, Derived classes may hold the data in any user-specific
 * format, and adapt to opengv-types.
 */
template<typename Tp>
class PoseAdapterBase
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	typedef Matrix<Tp, 3, 1> Vector3;
	typedef Sophus::SO3<Tp> SO3_T;

	typedef Matrix<Tp, 3, 1> Point3;

	/**
   * \brief Constructor.
   */
  PoseAdapterBase() :
      _t_w(Vector3::Zero()),
	  _cx(0), _cy(0){};
  /**
   * \brief Constructor.
   * \param[in] R A prior or known value for the rotation from the viewpoint
   *              to the world frame.
   */
  PoseAdapterBase( const SO3_T& R ) :
      _t_w(Vector3::Zero()),
      _R_cw(R),
	  _cx(0), _cy(0){};

  /**
   * \brief Constructor.
   * \param[in] t A prior or known value for the position of the viewpoint seen
   *              from the world frame.
   * \param[in] R A prior or known value for the rotation from the viewpoint
   *              to the world frame.
   */
  PoseAdapterBase(
      const Vector3 & t,
      const SO3_T & R ) :
      _t_w(t),
      _R_cw(R),
	  _cx(0), _cy(0){};

  /**
   * \brief Destructor.
   */
  virtual ~PoseAdapterBase() {};

  //Access of correspondences
  
  /**
   * \brief Retrieve the bearing vector of a correspondence.
   * \param[in] index The serialized index of the correspondence.
   * \return The corresponding bearing vector.
   */
  virtual Point3 getBearingVector(int index) const = 0;
  /**
   * \brief Retrieve the weight of a correspondence. The weight is supposed to
   *        reflect the quality of a correspondence, and typically is between
   *        0 and 1.
   * \param[in] index The serialized index of the correspondence.
   * \return The corresponding weight.
   */
  virtual Tp getWeight( int index ) const = 0;

  /**
   * \brief Retrieve the world point of a correspondence.
   * \param[in] index The serialized index of the correspondence.
   * \return The corresponding world point.
   */
  virtual Point3 getPointGlob(int index) const = 0;
  /**
   * \brief Retrieve the number of correspondences.
   * \return The number of correspondences.
   */
  virtual int getNumberCorrespondences() const = 0;

  //Access of priors or known values
  
  /**
   * \brief Retrieve the prior or known value for the position.
   * \return The prior or known value for the position.
   */
  Vector3 gettw() const { return _t_w; };
  /**
   * \brief Set the prior or known value for the position.
   * \param[in] t The prior or known value for the position.
   */
  void sett(const Vector3 & t) { _t_w = t; };
  /**
   * \brief Retrieve the prior or known value for the rotation.
   * \return The prior or known value for the rotation.
   */
  SO3_T getRcw() const { return _R_cw; };
  /**
   * \brief Set the prior or known value for the rotation.
   * \param[in] R The prior or known value for the rotation.
   */
  void setRcw(const SO3_T& R) { _R_cw = R; };

  void setFocal(const Tp fx, const Tp fy) { _fx = fx; _fy = fy; };

  Tp getFocal() const { return (_fx + _fy) /2; };


protected:
  /** The prior or known value for the translation from the world coordinate to the
   * camera coordinate system. Initialized to zero if not provided.
   */
	Vector3 _t_w;
  /** The prior or known value for the rotation from the WORLD coordinate system to the
   * camera coordinate system . Initialized to identity if not provided.
   */
	SO3_T _R_cw;
	/** The known camera internal parameters: focal length and principle points
	*/
	Tp _fx, _fy, _cx, _cy;

};


#endif 
