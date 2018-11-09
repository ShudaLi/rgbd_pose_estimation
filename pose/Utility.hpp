#ifndef BTL_OTHER_UTILITY_HELPER
#define BTL_OTHER_UTILITY_HELPER

//helpers based-on stl and boost

#include <iostream>
#include <fstream>
#include <vector>
#include <list>
#include <numeric>
#include <complex>
#include <string>
#include <chrono>
#include <memory>


#define SMALL 1e-50 // a small value
#define BTL_DOUBLE_MAX 10e20
// for print
template <class T>
std::ostream& operator << ( std::ostream& os, const std::vector< T > & v )
{
    os << "[";

    for ( typename std::vector< T >::const_iterator constItr = v.begin(); constItr != v.end(); ++constItr )
    {
        os << " " << ( *constItr ) << " ";
    }

    os << "]";
    return os;
}

template <class T>
std::ostream& operator << ( std::ostream& os, const std::list< T >& l_ )
{
    os << "[";
    for ( typename std::list< T >::const_iterator cit_List = l_.begin(); cit_List != l_.end(); cit_List++ )
    {
        os << " " << *cit_List << " ";
    }
    os << "]";
    return os;
}

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}
//calculate vector<> difference for testing
template< class T>
T matNormL1 ( const std::vector< T >& vMat1_, const std::vector< T >& vMat2_ )
{
    T tAccumDiff = 0;
    for(unsigned int i=0; i < vMat1_.size(); i++ )
    {
        T tDiff = vMat1_[i] - vMat2_[i];
        tDiff = tDiff >= 0? tDiff:-tDiff;
        tAccumDiff += tDiff;
    }
    return tAccumDiff;
}

template< class T >
void getNeighbourIdxCylinder(const unsigned short& usRows, const unsigned short& usCols, const T& i, std::vector< T >* pNeighbours_ )
{
    // get the neighbor 1d index in a cylindrical coordinate system
    int a = usRows*usCols;

    pNeighbours_->clear();
    pNeighbours_->push_back(i);
    T r = i/usCols;
    T c = i%usCols;
    T nL= c==0?        i-1 +usCols : i-1;    
    T nR= c==usCols-1? i+1 -usCols : i+1;
    pNeighbours_->push_back(nL);
    pNeighbours_->push_back(nR);

    if(r>0)//get up
    {
        T nU= i-usCols;
        pNeighbours_->push_back(nU);
        T nUL= nU%usCols == 0? nU-1 +usCols: nU-1;
        pNeighbours_->push_back(nUL);
        T nUR= nU%usCols == usCols-1? nU+1 -usCols : nU+1;
        pNeighbours_->push_back(nUR);
    }
    else if(r==usRows-1)//correspond to polar region
    {
        T t = r*usCols;
        for( T n=0; n<usCols; n++)
            pNeighbours_->push_back(t+n);
    }
    if(r<usRows-1)//get down
    {
        T nD= i+usCols;
        pNeighbours_->push_back(nD);
        T nDL= nD%usCols == 0? nD-1 +usCols: nD-1;
        pNeighbours_->push_back(nDL);
        T nDR= nD%usCols == usCols-1? nD+1 -usCols : nD+1;
        pNeighbours_->push_back(nDR);
    }

    return;
}


template <typename T>
std::vector<int> sortIndexes(const std::vector<T> &v) {

  // initialize original index locations
  std::vector<int> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  std::sort(idx.begin(), idx.end(), [&v](int i1, int i2) {return v[i1] > v[i2];});

  return idx;
}

using namespace std;
// places randomly selected  element at end of array
// then shrinks array by 1 element. Done when only 1 element is left
// m is the # to be selected from
// n is the total # of elements
template< class T >
class  RandomElements
{
    T* _idx;
    int _n;
public:
    RandomElements(int n):_n(n){
        _idx = new T[n];
    }
    ~RandomElements(){
        delete _idx;
    }

    void run( int m, vector< T >* p_v_idx_ )
    {
        p_v_idx_->clear();
        for (T i = 0; i < _n; i++) {
            _idx[i] = i;
        }
        int temp = 0;
        int ridx = _n-1;
        for(int j=(_n-1); j>_n-m-1; j--)// one pass through array
        {
            ridx = rand()%(j+1);// index = 0 to j
            temp = _idx[ridx];// value will be moved to end element
            _idx[ridx] = _idx[j];// end element value in random spot
            _idx[j] = temp;// selected element moved to end. This value is final
            p_v_idx_->push_back(temp);
        }
        return;
    }
};//class RandElement


// Prosac sampler used for PROSAC implemented according to "Matching with PROSAC
// - Progressive Sampling Consensus" by Chum and Matas.
template< class T >
class ProsacSampler {
public:
    ProsacSampler( const int min_num_samples, const int num_datapoints )
    : min_num_samples_(min_num_samples) 
    {
        num_datapoints_ = num_datapoints;
        ransac_convergence_iterations_ = 20000;
        kth_sample_number_ = 1;
    }
    ~ProsacSampler() {}

    // Set the sample such that you are sampling the kth prosac sample (Eq. 6).
    void SetSampleNumber(int k)
    {
        kth_sample_number_ = k;
    }

    // Samples the input variable data and fills the vector subset with the prosac
    // samples.
    // NOTE: This assumes that data is in sorted order by quality where data[i] is
    // of higher quality than data[j] for all i < j.
    bool Sample(std::vector<int>* subset_indices)
    {
        // Set t_n according to the PROSAC paper's recommendation.
        double t_n = ransac_convergence_iterations_;
        int n = this->min_num_samples_;
        // From Equations leading up to Eq 3 in Chum et al.
        for (int i = 0; i < this->min_num_samples_; i++) {
            t_n *= static_cast<double>(n - i) / (num_datapoints_ - i);
        }

        double t_n_prime = 1.0;
        // Choose min n such that T_n_prime >= t (Eq. 5).
        for (int t = 1; t <= kth_sample_number_; t++) {
            if (t > t_n_prime && n < num_datapoints_) {
                double t_n_plus1 = (t_n * (n + 1.0)) / (n + 1.0 - this->min_num_samples_);
                t_n_prime += ceil(t_n_plus1 - t_n);
                t_n = t_n_plus1;
                n++;
            }
        }
        subset_indices->reserve(this->min_num_samples_);
        if (t_n_prime < kth_sample_number_) {
            // Randomly sample m data points from the top n data points.
            std::vector<int> random_numbers;
            for (int i = 0; i < this->min_num_samples_; i++) {
                // Generate a random number that has not already been used.
                int rand_number;
                while (std::find(random_numbers.begin(),
                               random_numbers.end(),
                               (rand_number = rand()%(n) )) !=
                     random_numbers.end()) {
                }

                random_numbers.push_back(rand_number);

                // Push the *unique* random index back.
                subset_indices->push_back(rand_number);
            }
        } else {
            std::vector<int> random_numbers;
            // Randomly sample m-1 data points from the top n-1 data points.
            for (int i = 0; i < this->min_num_samples_ - 1; i++) {
              // Generate a random number that has not already been used.
              int rand_number;
              while (std::find(random_numbers.begin(),
                               random_numbers.end(),
                               (rand_number = rand()%(n-1) )) !=
                     random_numbers.end()) {
              }
              random_numbers.push_back(rand_number);

              // Push the *unique* random index back.
              subset_indices->push_back(rand_number);
            }
            // Make the last point from the nth position.
            subset_indices->push_back(n);
        }
        assert((int)subset_indices->size() == this->min_num_samples_);
        kth_sample_number_++;
        return true;
    }

    private:
    int num_datapoints_;
    // Number of iterations of PROSAC before it just acts like ransac.
    int ransac_convergence_iterations_;

    // The kth sample of prosac sampling.
    int kth_sample_number_;

    int min_num_samples_;
};


// A wrapper around the c++11 random generator utilities. This allows for a
// thread-safe random number generator that may be easily instantiated and
// passed around as an object.
template< class T >
class RandomNumberGenerator 
{
public:
    std::mt19937 util_generator;
    // Creates the random number generator using the current time as the seed.
    RandomNumberGenerator(){
        const unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        util_generator.seed(seed);
    }

    // Creates the random number generator using the given seed.
    RandomNumberGenerator(const unsigned seed){
        util_generator.seed(seed);
    }

    // Seeds the random number generator with the given value.
    void Seed(const unsigned seed){
        util_generator.seed(seed);
    }

    // Get a random double between lower and upper (inclusive).
    T RandFlt(const T lower, const T upper){
        std::uniform_real_distribution<T> distribution(lower, upper);
        return distribution(util_generator);
    }

    // Get a random double between lower and upper (inclusive).
    int RandInt(const int lower, const int upper){
        std::uniform_int_distribution<int> distribution(lower, upper);
        return distribution(util_generator);
    }

    // Generate a number drawn from a gaussian distribution.
    T RandGaussian(const T mean, const T std_dev){
        std::normal_distribution<T> distribution(mean, std_dev);
        return distribution(util_generator);
    }

    inline T Rand(const T lower, const T upper) {
        return RandFlt(lower, upper);
    }

    // Sets an Eigen type with random values between -1.0 and 1.0. This is meant
    // to replace the Eigen::Random() functionality.
    void SetRandom(Eigen::MatrixBase<T>* b) {
        for (int r = 0; r < b->rows(); r++) {
            for (int c = 0; c < b->cols(); c++) {
            (*b)(r, c) = Rand(-1.0, 1.0);
            }
        }
    }
};


// template< class T >
// class ProsacRandomElements: public RandElement< int >
// {
// public:
//     T _T_n;
//     T _T_prime_n;
//     T _n;
//     T _n_star;
//     int _T_N; // maximum iterations
//     int _m;// minimum pairs of data for estimating a pose
//     int _N;// size of all pairs 
//     int _t;
//     int _k_n_star;
//     ProsacRandomElements(vector< int >& sortedIdx_, const int T_N_ = 1000, const int m_){
//         _T_N = T_N_;
//         _N = (int)sortedIdx_.size();
//         _m = m_;
//         _T_n = _T_N;
//         for (int i = 0; i < _m; ++i)
//             _T_n = tatic_cast<float> (_m - i) / static_cast<float> (_N - i);
//         _T_prime_n = 1.0f;
//         _n = static_cast<float> (m);

//         // Define the n_Start coefficients from Section 2.2
//         _n_star = static_cast<float> (_N);

//         _t = 0;
//     }
//     ~ProsacRandomElements(){
//     }    

//     void run(vector< int > p_v_idx_ ){
//         if( _t < k_n_star){

//         }

//     }
// };




#endif
