rgbd_pose_estimation (Version 1.0 - 13th of May 2015)
---------------------------------------
Coming soon...


Release Notes:
--------------
* A header only library with example files to deliver moer accurate, precise and robust pose estimation than using traditional Perspective-n-Point approaches. 
* Tested on Windows and Linux 
  
Build Notes:
------------
Windows:

	Requirements:
		1) Microsoft Visual C++
		   From: https://support.microsoft.com/en-us/kb/2977003/
		2) Boost
		   From: http://www.boost.org/
		3) Eigen
		   From: http://eigen.tuxfamily.org/index.php?title=Main_Page
		   
	Optional requirements:
		1) OpenCV 3.0 Beta (To build the example)
		   From: http://opencv.org/
	    2) CMake and CMake-GuI (To create project files)
		   From: http://www.cmake.org/
	
    Building rgbd_pose_estimation:
	    As the library is composed of a bunch of headers of C++, no compilation is needed.
	
	Building the example
	    1) Install the OpenCV 3.0 Library 
		2) Create a folder 'build' 
		3) Run CMake-Gui and specify the path to dependences. 
		4) Press 'configure' and then 'generate' button
		5) Open the 'RGBDPoseEstimation.sln' and compile it.
		
    
	   