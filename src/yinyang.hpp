#ifndef YINYANG_HPP
#define YINYANG_HPP
#include "common.hpp"
#include "lloyd.hpp"
#include "kmeans_plus_plus.hpp"
#include <type_traits>
#include <vector>
#include <set>
#include <iostream>
#include <cmath>
#include <numeric>


// first iteration of yinyangkmeans
// ran when cluster center are just generated
// divide center into groups and initialize other values
template <class T>
int yinyang_first_iteration(const DataMat<T> &data, ClusterVec &cluster, CenterMat<T> &center, int G, double precision, ClusterVec &group, std::vector<std::vector<float>> &lbg, std::vector<float> &ub, CenterMat<T> &center_sum, ClusterVec &center_count, std::vector<std::set<int>> &centers_in_group, int D) {
	std::cerr << "start yinyang first iteration" << std::endl;
	int N = data.size();
	int K = center.size();
	CenterMat<T> center_center(G, T(D));
	kmeans_plus_plus_initialize(center, center_center);
	lloyd(center, group, center_center, precision, D, 5, false);
	/* run lloyd kmeans for the first iteration */
	CenterMat<T> workspace1(K, T(D));
	ClusterVec workspace2(K);
	lloyd_update_cluster(data, cluster, center);
	lloyd_update_center(data, cluster, center, precision, workspace1, workspace2);
	lloyd_update_cluster(data, cluster, center);
	std::cerr << "step 0 loss " << compute_loss(data, cluster, center) << std::endl;
	std::vector<std::vector<double>> d(N, std::vector<double>(K));
	for (int n = 0; n < N; ++n)
		for (int k = 0; k < K; ++k)
			d[n][k] = bvyyKMeansDistance(data[n], center[k]);
	/* initialize ub, lbg */
	for (int n = 0; n < N; ++n) {
		//std::cerr << "d.col(n) " << d.col(n) << std::endl;
		double mv = d[n][0];
		int mp = 0;
		for (int k = 1; k < K; ++k)
			if (d[n][k] < mv) {
				mv = d[n][k];
				mp = k;
			}
		ub[n] = mv;
		cluster[n] = mp;
	}
	for (int n = 0; n < N; ++n)
		for (int k = 0; k < K; ++k)
			if (k != cluster[n]) {
				int g = group[k];
				double tmp_d = d[n][k];
				float tmp_lbg = lbg[n][g];
				if (tmp_lbg < 0 || tmp_d < tmp_lbg)
					lbg[n][g] = tmp_d;
			} else {
				center_count[k] += 1;
				center_sum[k] += data[n];
			}
	for (int k = 0; k < K; ++k)
		centers_in_group[group[k]].insert(k);
	std::cerr << "finished yinyang first iteration" << std::endl;
	return 0;
}

template <class T, bool blocked>
bool update_center(CenterMat<T> &center, ClusterVec &group, CenterMat<T> &center_sum, ClusterVec &center_count, std::vector<float> &delta_c, std::vector<float> &delta_g, double precision, std::vector<double> &norm_center, std::vector<T> &block_center, const int B, const int D) {
	//std::cerr << "start update_center" << std::endl;
	bool changed = false;
	T tmp_center;
	const int K = center.size();
	const int G = delta_g.size();
	for (int g = 0; g < G; ++g)
		delta_g[g] = -1;
	for (int k = 0; k < K; ++k) {
		int cc = center_count[k];
		if (!cc)
			continue;
		tmp_center = center[k];
		center[k] = center_sum[k] / cc;
		float dc = bvyyKMeansDistance(center[k], tmp_center);
		if (dc > precision) {
			changed = true;
			if (blocked) {
				norm_center[k] = bvyyKMeansNorm(center[k]);
				generate_block_vector(block_center[k], center[k], B, D);
			}
		}
		delta_c[k] = dc;
		int g = group[k];
		float dg = delta_g[g];
		if (dg < dc)
			delta_g[g] = dc;
	}
	//std::cerr << "finished update_center" << std::endl;
	return changed;
}

template <class T, bool blocked>
bool yinyang_update_cluster(const DataMat<T> &data, ClusterVec &cluster, CenterMat<T> &center, ClusterVec &group, std::vector<std::set<int>> &centers_in_group, std::vector<std::vector<float>> &lbg, std::vector<float> &ub, std::vector<float> &delta_c, std::vector<float> &delta_g, CenterMat<T> &center_sum, std::vector<int> &center_count, std::vector<int> &count) {
	//std::cerr << "start yinyang_update_cluster" << std::endl;
	bool changed = false;
	int N = data.size();
	int G = lbg[0].size();
	for (int n = 0; n < N; ++n) {
		ub[n] += delta_c[cluster[n]];
		std::vector<float> old_lbg = lbg[n];
		for (int g = 0; g < G; ++g) {
			if (lbg[n][g] < 0)
				continue;
			lbg[n][g] = lbg[n][g] - delta_g[g];
			if (lbg[n][g] < 0)
				lbg[n][g] = 0;
		}
		/* Global filtering */
		float min_lbg = -1;
		for (int g = 0; g < G; ++g) {
			double tmp_lbg = lbg[n][g];
			if (tmp_lbg < 0)
				continue;
			if (min_lbg < 0 || min_lbg > tmp_lbg)
				min_lbg = tmp_lbg;
		}
		if (min_lbg < 0 || min_lbg > ub[n])
			continue;
		ub[n] = bvyyKMeansDistance(data[n], center[cluster[n]]);
		++count[n];
		if (min_lbg > ub[n])
			continue;
		//std::cerr << "pass global filtering ";
		/* Group filtering */
		for (int g = 0; g < G; ++g) {
			if (lbg[n][g] < 0 || lbg[n][g] > ub[n])
				continue;
			lbg[n][g] = -1;
			//std::cerr << "pass group filtering ";
			/* Local filtering */
			for (int c:centers_in_group[g]) {
				//std::cerr << "start local filtering ";
		//std::cerr << "test1" << std::endl;
				if (c == cluster[n])
					continue;
		//std::cerr << "test2" << std::endl;
				if (lbg[n][g] >= 0 && old_lbg[g] - delta_c[c] >= lbg[n][g])
					continue;
				if (blocked) {
					double tmp_b = bvyyKmeansLBB(n, norm_data, block_data, g, norm_center, block_center);
					if (tmp_b >= ub[n]) {
						if (lbg[n][g] < 0 || tmp_b < lbg[n][g])
							lbg[n][g] = tmp_b;
						continue;
					}
				}
				float tmp_d = bvyyKMeansDistance(data[n], center[c]);
				++count[n];
				if (tmp_d < ub[n]) {
					//std::cerr << "pass local filtering ";
					changed = true;
					int l = cluster[n];
					float tmp_lbg = lbg[n][group[l]];
					if (tmp_lbg < 0 || tmp_lbg > ub[n])
						lbg[n][group[l]] = ub[n];
					cluster[n] = c;
					ub[n] = tmp_d;
					center_count[l] -= 1;
					center_count[c] += 1;
					center_sum[l] -= data[n];
					center_sum[c] += data[n];
				} else {
					if (lbg[n][g] < 0 || tmp_d < lbg[n][g])
						lbg[n][g] = tmp_d;
				}
			}
		}
	}
	//std::cerr << "finished yinyang_update_cluster" << std::endl;
	return changed;
}

template <class T>
int cmp_center(CenterMat<T> &center, CenterMat<T> &center1, int D, double precision) {
	int K = center.size();
	for (int k = 0; k < K; ++k) {
		for (int d = 0; d < D; ++d)
			if (std::abs(center[k][d] - center1[k][d]) > precision) {
				return 1;
			}
	}

	return 0;
}

int cmp_cluster(ClusterVec &cluster, ClusterVec &cluster1) {
	int N = cluster.size();
	for (int n = 0; n < N; ++n)
		if (cluster[n] != cluster1[n])
			return 1;
	return 0;
}

template <class T, bool blocked>
int yinyang(const DataMat<T> &data, ClusterVec &cluster, CenterMat<T> &center, const int D, const int G, const int B, double precision, int max_iteration, bool until_converge, bool debug, const std::vector<double> &norm_data, const std::vector<T> &block_data, std::vector<double> &norm_center, std::vector<T> &block_center) {
	int N = data.size();
	int K = center.size();

	std::vector<std::set<int>> centers_in_group(G);
	ClusterVec group(K);		// which group a cluster center is in
	std::vector<std::vector<float>> lbg(N, std::vector<float>(G, -1));		// lower bound for group
	std::vector<float> delta_c(K);		// delta change of cluster center
	std::vector<float> delta_g(G);		// max delta change of cluster group
	std::vector<float> ub(N);		// upper bound for d(x, b(x))
	CenterMat<T> center_sum(K, T(D));	// sum of all points in certain cluster
	ClusterVec center_count(K);	// how many points in a cluster

	ClusterVec cluster1;
	CenterMat<T> center1;
	CenterMat<T> workspace1;
	ClusterVec workspace2;
	if (debug) {
		cluster1.resize(N);
		center1 = CenterMat<T>(K, T(D));
		workspace1 = CenterMat<T>(N, T(D));
		workspace2.resize(K);
	}
	yinyang_first_iteration<T, blocked>(data, cluster, center, G, precision, group, lbg, ub, center_sum, center_count, centers_in_group, D, norm_data, block_data, norm_center, block_center);
	//std::cerr << cluster;
	//std::cerr << "max iteration " << max_iteration << std::endl;
	double ll = compute_loss(data, cluster, center), nl;
	std::vector<int> count(N);
	for (int it = 1; it < max_iteration; ++it) {
		bool changed1, changed2;
		changed1 = update_center<T, blocked>(center, group, center_sum, center_count, delta_c, delta_g, precision, norm_center, block_center, B, D);
		//std::cerr << "center changed " << changed1 << std::endl;
		nl = compute_loss(data, cluster, center);
		if (nl - ll > 1) {
			std::cerr << "loss increase in update center in step " << it << std::endl;
		}
		ll = nl;
		if (debug) {
			lloyd_update_center<T, blocked>(data, cluster, center1, precision, workspace1, workspace2, norm_center, block_center, B, D);
			if (cmp_center(center, center1, D, precision)) {
				std::cerr << "different center in step " << it << std::endl;
			}
		}
		changed2 = yinyang_update_cluster<T, blocked>(data, cluster, center, group, centers_in_group, lbg, ub, delta_c, delta_g, center_sum, center_count, count, norm_data, block_data, norm_center, block_center);
		//std::cerr << "cluster changed " << changed2 << std::endl;
		nl = compute_loss(data, cluster, center);
		if (nl - ll > 1) {
			std::cerr << "loss increase in update cluster in step " << it << std::endl;
		}
		ll = nl;
		if (debug) {
			lloyd_update_cluster<T, blocked>(data, cluster1, center, norm_data, block_data, norm_center, block_center);
			if (cmp_cluster(cluster, cluster1)) {
				std::cerr << "different cluster in step " << it << std::endl;
			}
		}
		if (!changed1 && !changed2) {
			std::cerr << "converges at step " << it << std::endl;
			break;
		}
		if (until_converge)
			max_iteration += 1;
		nl = compute_loss(data, cluster, center);
		if (nl - ll > 1) {
			std::cerr << "loss increase in step " << it << std::endl;
		}
		int total_count = std::accumulate(count.begin(), count.end(), 0);
		std::cerr << "step " << it << " loss " << nl << " total number of distance count " <<  total_count << " percentage " << static_cast<double>(total_count)/static_cast<double>(N * K) << std::endl;
		ll = nl;
		if (until_converge)
			++max_iteration;
		std::fill(count.begin(), count.end(), 0);
	}

	return 0;
}

#endif
