#include "block.hpp"
#include <cmath>

inline double bvyyKMeansLBC(int n, const std::vector<double> &norm_data, int k, const std::vector<double> &norm_center) {
	double tmp1 = norm_data[n];
	double tmp2 = norm_center[k];
	return std::sqrt(tmp1*tmp1+tmp2*tmp2-2*tmp1*tmp2);
}

