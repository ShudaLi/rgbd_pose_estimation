
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

#define data_type float

void test_3d_3d(){
    cout << "%test_3d_3d()" << endl;
    const Matrix<data_type, 3, 1> t = generate_random_translation_uniform<data_type>(5.0);
    const Sophus::SO3<data_type> R = generate_random_rotation<data_type>(M_PI / 2, false);

    //cout << t << endl;
    //cout << R << endl;

    //derive correspondences based on random point-cloud
    int total = 100;

    data_type  or_3d = 0.1; //outlier
    data_type  n3d = 0.02; //3-D noise

    int iteration = 1000;
    int updated_iter = iteration;
    int test_n = 100;
    string noise_model = "Gaussian";

    data_type thre_3d = 0.06; //meter

    data_type min_depth = 0.4;
    data_type max_depth = 8.; //for Gaussian or uniform noise
    data_type f = 585.;
    data_type confidence = 0.99999;
    Matrix<data_type, -1, -1> e_s(2, test_n);
    Matrix<data_type, -1, -1> e_l(2, test_n);

    cout << "total = " << total << endl;
    cout << "noise_thre_3d = " << thre_3d << endl;
    cout << "outlier ratio = " << or_3d << endl;
    cout << "3d noise= " << n3d << endl;

    for (int jj = 0; jj < test_n; jj++){
        Matrix<data_type, Dynamic, Dynamic> Q, P;
        Matrix<data_type, Dynamic, Dynamic> all_weights(total, 3);
        if (!noise_model.compare("Uniform")){
            simulate_3d_3d_correspondences<data_type>(R, t, total, n3d, or_3d,  min_depth, max_depth, f, false,
                &Q, &P, &all_weights);
        }
        else if (!noise_model.compare("Gaussian")){
            simulate_3d_3d_correspondences<data_type>(R, t, total, n3d, or_3d,  min_depth, max_depth, f, true,
                &Q, &P, &all_weights);
        }

        AOOnlyPoseAdapter<data_type> adapter(P, Q);
        adapter.setFocal(f, f);
        //estimate camera pose using shinji_ransac
        cout << adapter.getMaxVotes() << endl;
        updated_iter = iteration;
        shinji_ransac2<data_type>(adapter, thre_3d, updated_iter, confidence);
        e_s.col(jj) = calc_percentage_err<data_type>(R, t, &adapter);
        shinji_ls1<data_type>(adapter);
        e_l.col(jj) = calc_percentage_err<data_type>(R, t, &adapter);

    }

    cout << "t" << "_s =[" << e_s.row(0) << "]';" << endl;
    cout << "t" << "_s =[" << e_l.row(0) << "]';" << endl;
    cout << "r" << "_s =[" << e_s.row(1) << "]';" << endl;
    cout << "r" << "_s =[" << e_l.row(1) << "]';" << endl;
    cout << endl;

    return;
}

void test_3d_2d(){
    cout << "%test_3d_2d()" << endl;
    const Matrix<data_type, 3, 1> t = generate_random_translation_uniform<data_type>(5.0);
    const Sophus::SO3<data_type> R = generate_random_rotation<data_type>(M_PI / 2, false);

    //cout << t << endl;
    //cout << R << endl;

    //derive correspondences based on random point-cloud
    int total = 100;

    data_type  or_3d = 0.1; //outlier
    data_type  n3d = 0.02; //3-D noise

    int iteration = 1000;
    int updated_iter = iteration;
    int test_n = 100;
    string noise_model = "Gaussian";

    data_type thre_3d = 0.06; //meter
    data_type thre_2d = 0.02;//1-cos alpha

    data_type min_depth = 0.4;
    data_type max_depth = 8.; //for Gaussian or uniform noise
    data_type f = 585.;
    data_type confidence = 0.99999;
    Matrix<data_type, -1, -1> e_s(2, test_n);
    Matrix<data_type, -1, -1> e_l(2, test_n);

    cout << "total = " << total << endl;
    cout << "noise_thre_3d = " << thre_3d << endl;
    cout << "outlier ratio = " << or_3d << endl;
    cout << "3d noise= " << n3d << endl;

    for (int jj = 0; jj < test_n; jj++){
        Matrix<data_type, Dynamic, Dynamic> Q, P, U;
        Matrix<data_type, Dynamic, Dynamic> all_weights(total, 3);
        if (!noise_model.compare("Uniform")){
            simulate_2d_3d_correspondences<data_type>(R, t, total, n3d, or_3d, min_depth, max_depth, f, false,
                &Q, &U, &P, &all_weights);
        }
        else if (!noise_model.compare("Gaussian")){
            simulate_2d_3d_correspondences<data_type>(R, t, total, n3d, or_3d, min_depth, max_depth, f, true,
                &Q, &U, &P, &all_weights);
        }

        PnPPoseAdapter<data_type> adapter(U,Q);
        adapter.setFocal(f, f);
        adapter.setWeights(all_weights);
        //estimate camera pose using shinji_ransac
        cout << adapter.getMaxVotes() << endl;
        updated_iter = iteration;
        kneip_prosac<data_type>(adapter, thre_2d, updated_iter, confidence);
        e_s.col(jj) = calc_percentage_err<data_type>(R, t, &adapter);
    }

    cout << "t" << "_s =[" << e_s.row(0) << "]';" << endl;
    cout << "r" << "_s =[" << e_s.row(1) << "]';" << endl;
    cout << endl;

    return;
}


void test_prosac(){
    ProsacSampler<data_type> ps(4, 100);
    for (int i = 0; i < 1000; ++i)
    {
        vector<int> select;
        ps.sample(&select);
        cout << i << " ";
        for(int j=0; j < (int)select.size(); j++)
            cout << select[j] << " ";
        cout << endl;
    }
}

int main()
{
    // test_prosac();
    test_3d_2d();
    // test_3d_3d();

    return 0;
}

