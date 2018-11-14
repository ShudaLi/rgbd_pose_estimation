
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
    data_type  n3d = 0.1; //3-D noise

    int iteration = 100000;
    int updated_iter = iteration;
    int test_n = 10;
    string noise_model = "Gaussian";

    data_type thre_3d = 0.25; //meter

    data_type min_depth = 0.4;
    data_type max_depth = 8.; //for Gaussian or uniform noise
    data_type f = 585.;
    data_type confidence = 0.9999;
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
        adapter.setWeights(all_weights);
        //estimate camera pose using shinji_ransac
        updated_iter = iteration;
        shinji_prosac<data_type>(adapter, thre_3d, updated_iter, confidence);
        cout << "prosac max " << adapter.getMaxVotes() << endl;
        cout << "prosac it " << updated_iter << endl;
        shinji_ls1<data_type>(adapter);
        e_s.col(jj) = calc_percentage_err<data_type>(R, t, &adapter);

        updated_iter = iteration;
        shinji_ransac2<data_type>(adapter, thre_3d, updated_iter, confidence);
        cout << "ransac max " << adapter.getMaxVotes() << endl;
        cout << "ransac it " << updated_iter << endl;
        cout << endl;
        shinji_ls1<data_type>(adapter);
        e_l.col(jj) = calc_percentage_err<data_type>(R, t, &adapter);
    }

    cout << "prosac t" << "_s =[" << e_s.row(0) << "]';" << endl;
    cout << "prosac r" << "_s =[" << e_s.row(1) << "]';" << endl;
    cout << "ransac t" << "_l =[" << e_l.row(0) << "]';" << endl;
    cout << "ransac r" << "_l =[" << e_l.row(1) << "]';" << endl;
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
    data_type  n2d = 0.1; //3-D noise

    int iteration = 100000;
    int updated_iter = iteration;
    int test_n = 10;
    string noise_model = "Gaussian";

    data_type thre_2d = 0.02;//1-cos alpha

    data_type min_depth = 0.4;
    data_type max_depth = 8.; //for Gaussian or uniform noise
    data_type f = 585.;
    data_type confidence = 0.8;
    Matrix<data_type, -1, -1> e_s(2, test_n);
    Matrix<data_type, -1, -1> e_l(2, test_n);

    cout << "total = " << total << endl;
    cout << "noise_thre_2d = " << thre_2d << endl;
    cout << "outlier ratio = " << or_3d << endl;
    cout << "3d noise= " << n2d << endl;

    for (int jj = 0; jj < test_n; jj++){
        Matrix<data_type, Dynamic, Dynamic> Q, P, U;
        Matrix<data_type, Dynamic, Dynamic> all_weights(total, 3);
        if (!noise_model.compare("Uniform")){
            simulate_2d_3d_correspondences<data_type>(R, t, total, n2d, or_3d, min_depth, max_depth, f, false,
                &Q, &U, &P, &all_weights);
        }
        else if (!noise_model.compare("Gaussian")){
            simulate_2d_3d_correspondences<data_type>(R, t, total, n2d, or_3d, min_depth, max_depth, f, true,
                &Q, &U, &P, &all_weights);
        }

        PnPPoseAdapter<data_type> adapter(U,Q);
        adapter.setFocal(f, f);
        adapter.setWeights(all_weights);
        //estimate camera pose using shinji_ransac
        updated_iter = iteration;
        kneip_prosac<data_type>(adapter, thre_2d, updated_iter, confidence);
        cout << "prosac max " << adapter.getMaxVotes() << endl;
        cout << "prosac it " << updated_iter << endl;
        e_s.col(jj) = calc_percentage_err<data_type>(R, t, &adapter);

        updated_iter = iteration;
        kneip_ransac<data_type>(adapter, thre_2d, updated_iter, confidence);
        cout << "ransac max " << adapter.getMaxVotes() << endl;
        cout << "ransac it " << updated_iter << endl;
        cout << endl;
        e_l.col(jj) = calc_percentage_err<data_type>(R, t, &adapter);
    }

    cout << "t" << "_s =[" << e_s.row(0) << "]';" << endl;
    cout << "r" << "_s =[" << e_s.row(1) << "]';" << endl;

    cout << "t" << "_l =[" << e_l.row(0) << "]';" << endl;
    cout << "r" << "_l =[" << e_l.row(1) << "]';" << endl;
    cout << endl;

    return;
}


void test_3d_3d_2d(){
    cout << "%test_3d_3d_2d()" << endl;
    const Matrix<data_type, 3, 1> t = generate_random_translation_uniform<data_type>(5.0);
    const Sophus::SO3<data_type> R = generate_random_rotation<data_type>(M_PI / 2, false);

    //cout << t << endl;
    //cout << R << endl;

    //derive correspondences based on random point-cloud
    int total = 100;

    data_type  or_3d = 0.1; //outlier
    data_type  n3d = 0.1; //3-D noise
    data_type  n2d = 0.1; //3-D noise

    int iteration = 100000;
    int updated_iter = iteration;
    int test_n = 10;
    string noise_model = "Gaussian";

    data_type thre_3d = 0.25; //meter
    data_type thre_2d = 0.25; //meter

    data_type min_depth = 0.4;
    data_type max_depth = 8.; //for Gaussian or uniform noise
    data_type f = 585.;
    data_type confidence = 0.9999;
    Matrix<data_type, -1, -1> e_l(2, test_n);
    Matrix<data_type, -1, -1> e_s(2, test_n);
    Matrix<data_type, -1, -1> e_k(2, test_n);
    Matrix<data_type, -1, -1> e_sk(2, test_n);

    cout << "total = " << total << endl;
    cout << "noise_thre_3d = " << thre_3d << endl;
    cout << "outlier ratio = " << or_3d << endl;
    cout << "3d noise= " << n3d << endl;

    for (int jj = 0; jj < test_n; jj++){
        Matrix<data_type, Dynamic, Dynamic> Q, P, U;
        Matrix<data_type, Dynamic, Dynamic> all_weights(total, 3);
        if (!noise_model.compare("Uniform")){
            simulate_2d_3d_3d_correspondences<data_type>(R, t, total, n2d, n3d, or_3d, min_depth, max_depth, f, false,
                &Q, &U, &P, &all_weights);
        }
        else if (!noise_model.compare("Gaussian")){
            simulate_2d_3d_3d_correspondences<data_type>(R, t, total, n2d, n3d, or_3d, min_depth, max_depth, f, true,
                &Q, &U, &P, &all_weights);
        }

        AOPoseAdapter<data_type> adapter(U, P, Q); //bv, Xg, Xc
        adapter.setFocal(f, f);
        adapter.setWeights(all_weights);
        //estimate camera pose using shinji_ransac
        updated_iter = iteration;
        shinji_kneip_prosac<data_type>(adapter, thre_3d, thre_2d, updated_iter, confidence);
        cout << "sk prosac max " << adapter.getMaxVotes() << endl;
        cout << "sk prosac it " << updated_iter << endl;
        shinji_ls<data_type>(adapter);
        e_s.col(jj) = calc_percentage_err<data_type>(R, t, &adapter);

        updated_iter = iteration;
        shinji_kneip_ransac<data_type>(adapter, thre_3d, thre_2d, updated_iter, confidence);
        cout << "sk ransac max " << adapter.getMaxVotes() << endl;
        cout << "sk ransac it " << updated_iter << endl;
        cout << endl;
        shinji_ls<data_type>(adapter);
        e_l.col(jj) = calc_percentage_err<data_type>(R, t, &adapter);

        // updated_iter = iteration;
        // shinji_ransac<data_type>(adapter, thre_3d, updated_iter, confidence);
        // cout << "shinji ransac max " << adapter.getMaxVotes() << endl;
        // cout << "shinji ransac it " << updated_iter << endl;
        // cout << endl;
        // shinji_ls<data_type>(adapter);
        // e_s.col(jj) = calc_percentage_err<data_type>(R, t, &adapter);

        // updated_iter = iteration;
        // kneip_ransac<data_type>(adapter, thre_2d, updated_iter, confidence);
        // cout << "kneip ransac max " << adapter.getMaxVotes() << endl;
        // cout << "kneip ransac it " << updated_iter << endl;
        // cout << endl;
        // e_k.col(jj) = calc_percentage_err<data_type>(R, t, &adapter);
    }

    cout << "sk prosac t" << "_s =[" << e_s.row(0) << "]';" << endl;
    cout << "sk prosac r" << "_s =[" << e_s.row(1) << "]';" << endl;
    // cout << "kneip ransac t" << "_s =[" << e_k.row(0) << "]';" << endl;
    // cout << "kneip ransac r" << "_s =[" << e_k.row(0) << "]';" << endl;
    cout << "sk ransac t" << "_l =[" << e_l.row(0) << "]';" << endl;
    cout << "sk ransac r" << "_l =[" << e_l.row(1) << "]';" << endl;
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
    // test_3d_2d();
    // test_3d_3d();
    test_3d_3d_2d();

    return 0;
}

