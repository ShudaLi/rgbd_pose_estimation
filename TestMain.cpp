
#define _USE_MATH_DEFINES

#include <Eigen/Core>
#include "common/Converters.hpp"
#include "AbsoluteOrientation.hpp"
#include "AbsoluteOrientationNormal.hpp"
#include "PnPPoseAdapter.hpp"
#include "AOPoseAdapter.hpp"
#include "NormalAOPoseAdapter.hpp"
#include "P3P.hpp"
#include "MinimalSolvers.hpp"
#include "Simulator.hpp"
#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>


using namespace Eigen;
using namespace std;
using namespace cv;
using namespace btl::utility;

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
	vector<double> v_noise_2d;
	vector<double> v_noise_normal;
	vector<double> v_noise_3d;

	vector<double> v_or;
	vector<int> v_iter;
	double or_2d = 0.8;
	double or_3d = 0.8;
	double or_nl = 0.1;

	int iteration = 1000;
	int updated_iter = iteration;
	int test_n = 1;
	string noise_model;

	double thre_2d = 0.02;//1-cos alpha
	double thre_3d = 0.03; //meter
	double thre_nl = 0.01;

	double min_depth = 0.4;
	double max_depth = 8.; //for Gaussian or uniform noise
	//double max_depth = 3.; //for Kinect noise mode
	double f = 585.;


#ifdef __gnu_linux__
	cv::FileStorage cFSRead("../Parameters.yml", cv::FileStorage::READ);
#elif _WIN32
	cv::FileStorage cFSRead("..\\Parameters.yml", cv::FileStorage::READ);
#else 
#error "OS not supported!"
#endif
	if (!cFSRead.isOpened()) {
		std::cout << "Load test.yml failed.";
		return;
	}
	cFSRead["total"] >> total;
	cFSRead["outlier"] >> v_or;
	cFSRead["noise_2d"] >> v_noise_2d;
	cFSRead["noise_3d"] >> v_noise_3d;
	cFSRead["noise_normal"] >> v_noise_normal;
	cFSRead["iteration"] >> v_iter;
	cFSRead["thre_2d"] >> thre_2d;
	cFSRead["thre_3d"] >> thre_3d;
	cFSRead["normal_thre"] >> thre_nl;
	cFSRead["test_n"] >> test_n;
	cFSRead["noise_model"] >> noise_model;

	vector<vector<int> > combination;

	for (int a = 0; a < v_noise_3d.size(); a++)	{
		vector<int> sub;
		sub.push_back(a);
		sub.push_back(a);
		sub.push_back(a);
		combination.push_back(sub);
	}

	cout << "close all;" << endl;
	cout << "%total = " << total << endl;
	cout << "%noise_2d = " << v_noise_2d << " outlier = " << v_or << " noise_3d = " << v_noise_3d << " noise_normal = " << v_noise_normal << endl;
	cout << "%noise_thre_2d = " << thre_2d << " noise_thre_3d = " << thre_3d << " normal_thre = " << thre_nl << endl;
	MatrixXd e_nsk(2, test_n), e_sk(2, test_n), e_s(2, test_n), e_k(2, test_n), e_l(2, test_n);// m0.setZero(); m1.setZero(); m2.setZero(); m3.setZero();
	MatrixXd e_opt(2, test_n), e_ns(2, test_n), e_nk(2, test_n), e_dw(2, test_n);// m0.setZero(); m1.setZero(); m2.setZero(); m3.setZero();
	for (int cc = 0; cc < combination.size(); cc++)
	{
		for (int oo = 0; oo < v_or.size(); oo++)
		{
			for (int ii = 0; ii < v_iter.size(); ii++)
			{
				iteration = v_iter[ii];
				cout << "%iteration = " << iteration << endl;
				
				or_2d = or_3d = or_nl = v_or[oo];
				cout << "%outlier ratio = " << or_2d << endl;

				int idx_3d = combination[cc][0];
				int idx_2d = combination[cc][1];
				int idx_n = combination[cc][2];

				double n2d = v_noise_2d[idx_2d]; //2-D noise 
				double n3d = v_noise_3d[idx_3d]; //3-D noise
				double nnl = v_noise_normal[idx_n] / 180.*M_PI;//normal noise

				for (int jj = 0; jj < test_n; jj++){
					MatrixXd Q, P, U;
					MatrixXd M, N;
					Matrix<short, Dynamic, Dynamic> all_weights(total, 3);
					if (!noise_model.compare("Uniform")){
						simulate_2d_3d_nl_correspondences<double>(R, t, total, n2d, or_2d, n3d, or_3d, nnl, or_nl, min_depth, max_depth, f, false,
							&Q, &M, &P, &N, &U, &all_weights);
					}
					else if (!noise_model.compare("Gaussian")){
						simulate_2d_3d_nl_correspondences<double>(R, t, total, n2d, or_2d, n3d, or_3d, nnl, or_nl, min_depth, max_depth, f, true,
							&Q, &M, &P, &N, &U, &all_weights);
					}
					else if (!noise_model.compare("Kinect")){
						max_depth = 3.;
						simulate_kinect_2d_3d_nl_correspondences<double>(R, t, total, n2d, or_2d, or_3d, nnl, or_nl, min_depth, max_depth, f,
							&Q, &M, &P, &N, &U, &all_weights);
					}
					NormalAOPoseAdapter<double> adapter(U, P, N, Q, M);

					//convert to opencv data structure
					updated_iter = iteration;
					Mat m2D, m3D, cameraMatrix, inliers, rvec, tvec;
					cameraMatrix.create(3, 3, CV_64FC1); cameraMatrix.setTo(0.);
					cameraMatrix.at<double>(0, 0) = f;
					cameraMatrix.at<double>(1, 1) = f;
					cameraMatrix.at<double>(2, 2) = 1.;
					cvtCV(Q, U, f, &m3D, &m2D);
					cv::solvePnPRansac(m3D, m2D, cameraMatrix, Mat(), rvec, tvec, false, updated_iter, 8.f, 0.99f, inliers, cv::SOLVEPNP_EPNP);
					cv::solvePnPRansac(m3D, m2D, cameraMatrix, Mat(), rvec, tvec, true, updated_iter, 8.f, 0.99f, inliers, cv::SOLVEPNP_ITERATIVE);
					//convert to rotation matrix
					Mat cvmRw;	Rodrigues(rvec, cvmRw);
					Matrix3d RR;
					Vector3d tt;
					cvtEigen(cvmRw, tvec, &RR, &tt);
					adapter.setRcw(RR);
					adapter.sett(tt);
					e_l.col(jj) = calc_percentage_err<double>(R, t, &adapter);

					//estimate camera pose using kneip_ransac
					updated_iter = iteration;
					kneip_ransac<double>(adapter, thre_2d, updated_iter, 0.99);
					e_k.col(jj) = calc_percentage_err<double>(R, t, &adapter);

					//estimate camera pose using shinji_ransac
					updated_iter = iteration;
					shinji_ransac<double>(adapter, thre_3d, updated_iter, 0.99);
					e_s.col(jj) = calc_percentage_err<double>(R, t, &adapter);

					//estimate camera pose using shinji_kneip_ransac
					updated_iter = iteration;
					shinji_kneip_ransac<double>(adapter, thre_3d, thre_2d, updated_iter, 0.99);
					adapter.getInlierIdx();
					e_sk.col(jj) = calc_percentage_err<double>(R, t, &adapter);
						
					//estimate camera pose using nl_kneip_ransac
					updated_iter = iteration;
					nl_kneip_ransac<double>(adapter, thre_2d, thre_nl, updated_iter, 0.99);
					e_nk.col(jj) = calc_percentage_err<double>(R, t, &adapter);

					//estimate camera pose using nl_shinji_ransac
					updated_iter = iteration;
					nl_shinji_ransac<double>(adapter, thre_3d, thre_nl, updated_iter, 0.99);
					e_ns.col(jj) = calc_percentage_err<double>(R, t, &adapter);

					updated_iter = iteration;
					nl_shinji_kneip_ransac<double>(adapter, thre_3d, thre_2d, thre_nl, updated_iter, 0.99);
					e_nsk.col(jj) = calc_percentage_err<double>(R, t, &adapter);

					//estimate least square optimized solutions
					nl_shinji_kneip_ls<double>(adapter);
					e_opt.col(jj) = calc_percentage_err<double>(R, t, &adapter);

					//estimate using dynamic weights
					adapter.setWeights(all_weights);
					nl_shinji_kneip_ls<double>(adapter);
					e_dw.col(jj) = calc_percentage_err<double>(R, t, &adapter);
				}

				cout << endl;
				cout << "t" << cc << "_dw=[" << e_dw.row(0) << "]';" << endl;  //row(0) translation error
				cout << "t" << cc << "_opt=[" << e_opt.row(0) << "]';" << endl;
				cout << "t" << cc << "_nsk =[" << e_nsk.row(0) << "]';" << endl;
				cout << "t" << cc << "_ns=[" << e_ns.row(0) << "]';" << endl;
				cout << "t" << cc << "_nk=[" << e_nk.row(0) << "]';" << endl;
				cout << "t" << cc << "_sk =[" << e_sk.row(0) << "]';" << endl;
				cout << "t" << cc << "_s =[" << e_s.row(0) << "]';" << endl;
				cout << "t" << cc << "_k =[" << e_k.row(0) << "]';" << endl;
				cout << "t" << cc << "_l =[" << e_l.row(0) << "]';" << endl;
				cout << endl;
				cout << "r" << cc << "_dw=[" << e_dw.row(1) << "]';" << endl; //row(1) rotational error
				cout << "r" << cc << "_opt=[" << e_opt.row(1) << "]';" << endl;
				cout << "r" << cc << "_nsk =[" << e_nsk.row(1) << "]';" << endl;
				cout << "r" << cc << "_ns=[" << e_ns.row(1) << "]';" << endl;
				cout << "r" << cc << "_nk=[" << e_nk.row(1) << "]';" << endl;
				cout << "r" << cc << "_sk =[" << e_sk.row(1) << "]';" << endl;
				cout << "r" << cc << "_s =[" << e_s.row(1) << "]';" << endl;
				cout << "r" << cc << "_k =[" << e_k.row(1) << "]';" << endl;
				cout << "r" << cc << "_l =[" << e_l.row(1) << "]';" << endl;
				cout << endl;

				cout << "figure;" << endl;
				cout << "boxplot([t" << cc << "_dw, t" << cc << "_opt, t" << cc << "_nsk, t" << cc << "_ns, t" << cc << "_nk, t" << cc << "_sk, t" << cc << "_s, t" << cc << "_k, t" << cc << "_l] ," <<
				          "{ 'dw', 'opt', 'nl+s+k', 'nl+s', 'nl+k', 's+k', 's', 'k', 'l' }); " << endl;
				cout << "axis([0.5,8.5,0,30]);" << endl;
				cout << "xlabel('approaches') " << endl;
				cout << "ylabel('translational error (%)')" << endl;
				cout << "title('noise_{3d} = " << n3d << "; noise_{2d} = " << n2d << "; noise_{nl} = " << nnl << " " << noise_model << "')" << endl;
				cout << "set(gcf, 'Position', [0 0 200 240], 'PaperSize', [400 600]); " << endl;
				cout << "print('tfig" << cc << ".eps','-depsc');" << endl;
				cout << endl;
				cout << "figure;" << endl;
				cout << "boxplot([r" << cc << "_dw, r" << cc << "_opt, r" << cc << "_nsk, r" << cc << "_ns, r" << cc << "_nk, r" << cc << "_sk, r" << cc << "_s, r" << cc << "_k, t" << cc << "_l] ," << 
						"{'dw', 'op','nl+s+k','nl+s','nl+k','s+k','s','k', 'l'});" << endl;
				cout << "axis([0.5,8.5,0,30]);" << endl;
				cout << "xlabel('approaches') " << endl;
				cout << "ylabel('rotational error (%)')" << endl;
				cout << "title('noise_{3d} = " << n3d << "; noise_{2d} = " << n2d << "; noise_{nl} = " << nnl << " " << noise_model << "')" << endl;
				cout << "set(gcf, 'Position', [0 0 200 240], 'PaperSize', [400 600]); " << endl;
				cout << "print('rfig" << cc << ".eps','-depsc');" << endl;
				cout << endl;
			}

		}

	}

	cout << "SimpT = [";
	for (int cc = 0; cc < combination.size(); cc++)
	{
		//cout << "t" << cc << "_dw ";
		cout << "t" << cc << "_opt ";
		cout << "t" << cc << "_nsk ";
		cout << "t" << cc << "_s ";
		cout << "t" << cc << "_k ";
	}
	cout << "];" << endl;

	cout << "FullT = [";
	for (int cc = 0; cc < combination.size(); cc++)
	{
		//cout << "t" << cc << "_dw ";
		cout << "t" << cc << "_opt ";
		cout << "t" << cc << "_nsk ";
		cout << "t" << cc << "_ns ";
		cout << "t" << cc << "_nk ";
		cout << "t" << cc << "_sk ";
		cout << "t" << cc << "_s ";
		cout << "t" << cc << "_k ";
	}
	cout << "];" << endl;

	cout << "SimpR = [";
	for (int cc = 0; cc < combination.size(); cc++)
	{
		//cout << "t" << cc << "_dw ";
		cout << "r" << cc << "_opt ";
		cout << "r" << cc << "_nsk ";
		cout << "r" << cc << "_s ";
		cout << "r" << cc << "_k ";
	}
	cout << "];" << endl;

	cout << "FullR = [";
	for (int cc = 0; cc < combination.size(); cc++)
	{
		//cout << "r" << cc << "_dw ";
		cout << "r" << cc << "_opt ";
		cout << "r" << cc << "_nsk ";
		cout << "r" << cc << "_ns ";
		cout << "r" << cc << "_nk ";
		cout << "r" << cc << "_sk ";
		cout << "r" << cc << "_s ";
		cout << "r" << cc << "_k ";
	}
	cout << "];" << endl;
	return;
}



int main()
{

	test_all();
	

	return 0;
}

