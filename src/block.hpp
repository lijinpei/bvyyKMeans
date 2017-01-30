#ifndef _BLOCK_HPP
#define _BLOCK_HPP
#include "common.hpp"
#include "cmath"

template <class T>
void generate_block_vector(T &vec_target, const T&vec_source, const int B, const int D) {
	int div = D / B;
	int j1 = 0, j2 = 0;
	double tmp_v;
	for (int b = 0; b < B - 1; ++b) {
		tmp_v = 0;
		j2 += div;
		for (; j1 < j2; ++j1)
			tmp_v += vec_source[j1] * vec_source[j1];
		bvyyKMeansInsert(vec_target, b, tmp_v);
	}
	for (;j1 < D; ++j1)
		tmp_v += vec_source[j1] * vec_source[j1];
	bvyyKMeansInsert(vec_target, B - 1, tmp_v);
}

template <class T>
inline double bvyyKMeansLBB(int n, const std::vector<double> &norm_data, const std::vector<T> &block_data, int k, std::vector<double> &norm_center, const std::vector<T> &block_center) {
	double norm1 = norm_data[n];
	double norm2 = norm_center[k];
	const T& bv1 = block_data[n];
	const T& bv2 = block_center[k];
	double tmp = norm1*norm1 + norm2*norm2 - 2 * bvyyKMeansInnerProduct(bv1, bv2);
	if (tmp <= 0)
		return 0;
	else
		return std::sqrt(tmp);
}

inline double bvyyKMeansLBC(int n, const std::vector<double> &norm_data, int k, const std::vector<double> &norm_center) {
	double tmp1 = norm_data[n];
	double tmp2 = norm_center[k];
	return std::sqrt(tmp1*tmp1+tmp2*tmp2-2*tmp1*tmp2);
}

#endif
