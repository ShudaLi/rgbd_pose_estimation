#define _USE_MATH_DEFINES

#include <Eigen/Core>
#include "AbsoluteOrientation.hpp"
#include "AOOnlyPoseAdapter.hpp"
#include "Simulator.hpp"
#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace Eigen;

extern "C" {
 
	void ao(float* x_w_, float* x_c_, int n_, float* R_cw_, float* t_)
	{
        cout << "ao()" << endl;
        MatrixXf Xw = Map<MatrixXf >(x_w_, 3, n_);
        // cout << "Xw " << Xw << endl;
        MatrixXf Xc = Map<MatrixXf >(x_c_, 3, n_);
        // cout << "Xc " << Xc << endl;
        AOOnlyPoseAdapter<float> adapter(Xc, Xw);
        float f = 555.f;
        adapter.setFocal(f, f);

        // int updated_iter = 1000;
        // float thre_3d = 0.1;
        // float confidence = 0.99999f;
        // shinji_ransac2<float>(adapter, thre_3d, updated_iter, confidence);
        // cout << "updated_iter = " << updated_iter << endl;
        // cout << "inliers = " << adapter.getMaxVotes() << endl;
        shinji_ls2<float>(adapter);
        Matrix3f R_cw = adapter.getRcw().matrix();
        Matrix3f Rp = R_cw.transpose();
        float* data = Rp.data();
        for (int i=0; i < 9; i++)
            R_cw_[i] = data[i];

        Vector3f tw = adapter.gettw();
        for (int i=0; i < 3; i++)
            t_[i] = tw[i];
        // cout << SE3_[0] << endl;
	}

    void ao_ransac(float* x_w_, float* x_c_, int n_, float* R_cw_, float* t_)
    {
        cout << "ao()" << endl;
        MatrixXf Xw = Map<MatrixXf >(x_w_, 3, n_);
        // cout << "Xw " << Xw << endl;
        MatrixXf Xc = Map<MatrixXf >(x_c_, 3, n_);
        // cout << "Xc " << Xc << endl;
        AOOnlyPoseAdapter<float> adapter(Xc, Xw);
        float f = 555.f;
        adapter.setFocal(f, f);

        int updated_iter = 1000;
        float thre_3d = 0.1;
        float confidence = 0.99999f;
        shinji_ransac2<float>(adapter, thre_3d, updated_iter, confidence);
        cout << "updated_iter = " << updated_iter << endl;
        cout << "inliers = " << adapter.getMaxVotes() << endl;
        shinji_ls1<float>(adapter);
        Matrix3f R_cw = adapter.getRcw().matrix();
        Matrix3f Rp = R_cw.transpose();
        float* data = Rp.data();
        for (int i=0; i < 9; i++)
            R_cw_[i] = data[i];

        Vector3f tw = adapter.gettw();
        for (int i=0; i < 3; i++)
            t_[i] = tw[i];
        // cout << SE3_[0] << endl;
    }

    void py2c(float* array, int N)
    {
        for (int i=0; i<N; i++) 
            cout << array[i] << endl;
    }
}

