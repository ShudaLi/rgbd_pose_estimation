
#define _USE_MATH_DEFINES

#include <Eigen/Core>
#include "AbsoluteOrientation.hpp"
#include "AOOnlyPoseAdapter.hpp"
#include "Simulator.hpp"
#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>


using namespace Eigen;
using namespace std;
using namespace cv;
//using namespace btl::utility;

void cvtCV(const MatrixXd& Q_, const MatrixXd U_, const double& f_, Mat* m3D_, Mat* m2D_){
	m2D_->create(1, U_.cols(), CV_64FC2);
	m3D_->create(1, Q_.cols(), CV_64FC3);
	double* p2D = (double*)m2D_->data;
	double* p3D = (double*)m3D_->data;
	for (int c = 0; c < U_.cols(); c++)	{
		*p3D++ = Q_(0, c);
		*p3D++ = Q_(1, c);
		*p3D++ = Q_(2, c);

		*p2D++ = U_(0, c) / U_(2, c) *f_;
		*p2D++ = U_(1, c) / U_(2, c) *f_;
	}
	return;
}

void cvtEigen(const Mat& m, const Mat& t, Matrix3d* mm, Vector3d* tt) {
	for (int r = 0; r < m.rows; r++){
		for (int c = 0; c < m.cols; c++)
		{
			(*mm)(r, c) = m.at<double>(r, c);
		}
		(*tt)(r) = t.at<double>(r);
	}
	return;
}

void test_all(){
	cout << "%test_all()" << endl;
	const Vector3d t = generate_random_translation_uniform<double>(5.0);
	const Sophus::SO3<double> R = generate_random_rotation<double>(M_PI / 2, false);

	//cout << t << endl;
	//cout << R << endl;

	//derive correspondences based on random point-cloud
	int total = 100;

	double or_3d = 0.1;
	double n3d = 0.05; //3-D noise

	int iteration = 1000;
	int updated_iter = iteration;
	int test_n = 100;
	string noise_model = "Gaussian";

	double thre_3d = 0.06; //meter

	double min_depth = 0.4;
	double max_depth = 8.; //for Gaussian or uniform noise
	double f = 585.;
	double confidence = 0.99999;
	MatrixXd e_s(2, test_n);

	cout << "total = " << total << endl;
	cout << "noise_thre_3d = " << thre_3d << endl;
	cout << "outlier ratio = " << or_3d << endl;
	cout << "3d noise= " << n3d << endl;

	for (int jj = 0; jj < test_n; jj++){
		MatrixXd Q, P;
		Matrix<short, Dynamic, Dynamic> all_weights(total, 3);
		if (!noise_model.compare("Uniform")){
			simulate_3d_3d_correspondences<double>(R, t, total, n3d, or_3d,  min_depth, max_depth, f, false,
				&Q, &P, &all_weights);
		}
		else if (!noise_model.compare("Gaussian")){
			simulate_3d_3d_correspondences<double>(R, t, total, n3d, or_3d,  min_depth, max_depth, f, false,
				&Q, &P, &all_weights);
		}

		AOOnlyPoseAdapter<double> adapter(P, Q);
		adapter.setFocal(f, f);
		//estimate camera pose using shinji_ransac
		updated_iter = iteration;
		shinji_ransac2<double>(adapter, thre_3d, updated_iter, confidence);
		e_s.col(jj) = calc_percentage_err<double>(R, t, &adapter);
	}

	cout << "t" << "_s =[" << e_s.row(0) << "]';" << endl;
	cout << "r" << "_s =[" << e_s.row(1) << "]';" << endl;
	cout << endl;

	return;
}



int main()
{

	test_all();

	return 0;
}

