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
    : _m(min_num_samples) 
    {
        _N = num_datapoints;
        _T_N = 20000;
        _t = 1;
    }
    ~ProsacSampler() {}

    // Set the sample such that you are sampling the kth prosac sample (Eq. 6).
    void setSampleNumber(int k)
    {
        _t = k;
    }

    // Samples the input variable data and fills the vector subset with the prosac
    // samples.
    // NOTE: This assumes that data is in sorted order by quality where data[i] is
    // of higher quality than data[j] for all i < j.
    bool sample(std::vector<int>* subset_indices)
    {
        // Set t_n according to the PROSAC paper's recommendation.
        T t_n = _T_N;
        int n = this->_m;
        // From Equations leading up to Eq 3 in Chum et al.
        for (int i = 0; i < this->_m; i++) {
            t_n *= static_cast<T >(n - i) / (_N - i);
        }

        T t_n_prime = 1.0;
        // Choose min n such that T_n_prime >= t (Eq. 5).
        for (int t = 1; t <= _t; t++) {
            if (t > t_n_prime && n < _N) {
                T t_n_plus1 = (t_n * (n + 1.0)) / (n + 1.0 - this->_m);
                t_n_prime += ceil(t_n_plus1 - t_n);
                t_n = t_n_plus1;
                n++;
            }
        }
        subset_indices->reserve(this->_m);
        if (t_n_prime < _t) {
            // Randomly sample m data points from the top n data points.
            std::vector<int> random_numbers;
            for (int i = 0; i < this->_m; i++) {
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
            for (int i = 0; i < this->_m - 1; i++) {
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
        assert((int)subset_indices->size() == this->_m);
        _t++;
        return true;
    }

    private:
    int _N; // total number of data point
    int _T_N; // Number of iterations of PROSAC before it just acts like ransac.
    int _t; // The kth sample of prosac sampling.
    int _m; //minum number of samples
};


#endif
