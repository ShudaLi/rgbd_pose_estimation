cmake_minimum_required (VERSION 2.8)

if(COMMAND cmake_policy)
  cmake_policy(SET CMP0043 NEW)
  #cmake_policy(SET CMP0038 NEW)
  #cmake_policy(SET CMP0020 NEW)
endif()

project (RGBDPoseEstimation)

find_package( OpenCV REQUIRED )

if(UNIX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")
endif()

find_package(Eigen3 REQUIRED PATHS "c:/all_libs/eigen/include/eigen3")
include_directories ( ${EIGEN3_INCLUDE_DIR} )
include_directories ( "c:/all_libs/eigen/include/eigen3" )
include_directories ( ${PROJECT_SOURCE_DIR}/sophus )
include_directories ( ${PROJECT_SOURCE_DIR}/pose )
if( UNIX )

elseif( WIN32 )
message(STATUS "platform: Windows")
include_directories ( $ENV{OPENCV_INC} )
endif()

file(GLOB POSE_HEADERS "pose/[a-zA-Z]*.h" "pose/[a-zA-Z]*.hpp" )
message(STATUS "header names: ${POSE_HEADERS}")

add_library( rgbd_pose_estimation STATIC ${POSE_HEADERS} )
set_target_properties(rgbd_pose_estimation PROPERTIES LINKER_LANGUAGE CXX)


file(GLOB TESTER_SOURCE "TestMain.cpp" )
message(STATUS "header names: ${TESTER_SOURCE}")

add_executable( example ${TESTER_SOURCE} )
target_link_libraries( example  opencv_core  opencv_imgproc opencv_calib3d)

add_executable( simple SimpleMain.cpp )
target_link_libraries( simple  opencv_core  opencv_imgproc opencv_calib3d)

add_library( absolute SHARED Library.cpp )
target_link_libraries( absolute  opencv_core  opencv_imgproc opencv_calib3d)





