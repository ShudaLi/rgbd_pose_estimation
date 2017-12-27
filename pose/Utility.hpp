#ifndef BTL_OTHER_UTILITY_HELPER
#define BTL_OTHER_UTILITY_HELPER

//helpers based-on stl and boost

#include <iostream>
#include <fstream>
#include <vector>
#include <list>
#include <complex>
#include <string>


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



#endif
