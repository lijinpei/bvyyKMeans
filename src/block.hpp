#ifndef _BLOCK_HPP
#define _BLOCK_HPP
#include "common.hpp"
#include "cmath"

template <class T>
void generate_block_vector(T &vec_target, const T&vec_source, const int B, const int D) {
	int div = D / B;
	int j1 = 0, j2 = 0;
	double tmp_v = 0;
	for (int b = 0; b < B - 1; ++b) {
		tmp_v = 0;
		j2 += div;
		for (; j1 < j2; ++j1)
			tmp_v += vec_source[j1] * vec_source[j1];
		bvyyKMeansInsert(vec_target, b, std::sqrt(tmp_v));
	}
	for (;j1 < D; ++j1)
		tmp_v += vec_source[j1] * vec_source[j1];
	bvyyKMeansInsert(vec_target, B - 1, std::sqrt(tmp_v));
}

template <class T>
inline double bvyyKMeansLBB(double norm1, const T &bv1, double &norm2, const T &bv2) {
	double tmp = norm1*norm1 + norm2*norm2 - 2 * bvyyKMeansInnerProduct(bv1, bv2);
	if (tmp <= 0)
		return 0;
	else
		return std::sqrt(tmp);
}

inline double bvyyKMeansLBC(double norm1, double norm2) {
	return std::sqrt(norm1*norm1+norm2*norm2-2*norm1*norm2);
}

#endif
