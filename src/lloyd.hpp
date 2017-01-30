#ifndef LLOYD_HPP
#define LLOYD_HPP
#include "common.hpp"
#include "block.hpp"

template <class T, bool blocked>
bool lloyd_update_center(const DataMat<T> &data, const ClusterVec &cluster, CenterMat<T> &center, double precision, CenterMat<T> &workspace1, ClusterVec &workspace2, const int B, const int D, std::vector<double> &norm_center, std::vector<T> &block_center, std::vector<bool> &Y, std::vector<bool> &Z, const std::vector<double> &min_dist) {
	//std::cerr << "start lloyd update center" << std::endl;
	const int N = data.size();
	const int K = center.size();
	double l1 = compute_loss(data, cluster, center);
	bool changed = false;
	for (int k = 0; k < K; ++k)
		workspace1[k].clear();
	std::fill(workspace2.begin(), workspace2.end(), 0);
	if (blocked) {
		std::fill(Y.begin(), Y.end(), false);
		std::fill(Z.begin(), Z.end(), false);
	}
	for (int n = 0; n < N; ++n) {
		int c = cluster[n];
		workspace2[c] += 1;
		workspace1[c] += data[n];
	}
	for (int k = 0; k < K; ++k)
		if (0 == workspace2[k]) {
			std::cerr << "zero data point in cluster" << std::endl;
			continue;
		} else {
			T tmp_vec = workspace1[k] / workspace2[k];
			if (bvyyKMeansDistance(tmp_vec, center[k]) > precision)
				changed = true;
			else if (blocked)
					Y[k] = true;
			center[k] = tmp_vec;
		}
	if (blocked) {
		for (int n = 0; n < N; ++n)
			if (bvyyKMeansDistance(data[n], center[cluster[n]]) < min_dist[n])
				Z[n] = true;
		for (int k = 0; k < K; ++k) {
			norm_center[k] = bvyyKMeansNorm(center[k]);
			generate_block_vector(block_center[k], center[k], B, D);
		}
	}
	double l2 = compute_loss(data, cluster, center);
	if (l2 - l1 > 1)
		std::cerr << "Loss increases in update center" << std::endl;
	//std::cerr << "finished lloyd update center" << std::endl;
	return changed;
}

template <class T, bool blocked>
bool lloyd_update_cluster(const DataMat<T> &data, ClusterVec &cluster, const CenterMat<T> &center, const std::vector<double> &norm_data, const std::vector<T> &block_data, std::vector<double> &norm_center, std::vector<T> &block_center, const std::vector<bool> &Y, const std::vector<bool> Z, std::vector<double> &min_dist) {
	//std::cerr << "start lloyd update cluster" << std::endl;
	double l1 = compute_loss(data, cluster, center);
	bool changed = false;
	int N = data.size();
	int K = center.size();
	for (int n = 0; n < N; ++n) {
		//std::cerr << "lloyd update cluster iteration " << n << std::endl;
		int mp = 0;
		float mv = bvyyKMeansDistance(data[n], center[0]);
		for (int k = 1; k < K; ++k) {
			if (Z[n] && Y[k])
				continue;
			if (bvyyKMeansLBC(norm_data[n], norm_center[k]) >= mv)
				continue;
			if (bvyyKMeansLBB(norm_data[n], block_data[n], norm_center[k], block_center[k]) >= mv)
				continue;
			float nv = bvyyKMeansDistance(data[n], center[k]);
			if (nv < mv) {
				mp = k;
				mv = nv;
			}
		}
		if (cluster[n] != mp)
			changed = true;
	/*
		double d1 = bvyyKMeansDistance(data[n], center[mp]);
		double d2 = bvyyKMeansDistance(data[n], center[cluster[n]]);
		if (cluster[n] != mp && d1 > d2)
			std::cerr << "Loss increases in single update cluster " << mp << " " << cluster[n] << std::endl;
	*/
		cluster[n] = mp;
		if (blocked) {
			min_dist[n] = mv;
		}
	}
	double l2 = compute_loss(data, cluster, center);
	if (l2 > l1)
		std::cerr << "Loss increases in update cluster" << std::endl;
	//std::cerr << "finished lloyd update cluster" << std::endl;
	return changed;
}

template <class T, bool blocked>
int lloyd(const DataMat<T> &data, ClusterVec &cluster, CenterMat<T> &center, double precision, int D, int max_interation, bool until_converge, const int B, const std::vector<double> &norm_data, const std::vector<T> &block_data, std::vector<double> &norm_center, std::vector<T> &block_center) {
	std::cerr << "start lloyd iteration" << std::endl;
	const int K = center.size();
	const int N = data.size();

	CenterMat<T> workspace1(K, T(D));
	ClusterVec workspace2(K);
	double ll;
	double nl;
	std::vector<bool> Y, Z;
	std::vector<double> min_dist;
	if (blocked) {
		Y.resize(K);
		Z.resize(N);
		min_dist.resize(N);
		std::fill(Y.begin(), Y.end(), false);
		std::fill(Z.begin(), Z.end(), false);
	}
	lloyd_update_cluster<T, blocked>(data, cluster, center, norm_data, block_data, norm_center, block_center, Y, Z, min_dist);
	for (int i = 0; i < max_interation; ++i) {
		//std::cerr << "start iteration " << i << std::endl;
		bool changed1, changed2;
		changed1 = lloyd_update_center<T, blocked>(data, cluster, center, precision, workspace1, workspace2, B, D, norm_center, block_center, Y, Z, min_dist);
		//std::cerr << "center changed " << changed1 << std::endl;
		changed2 = lloyd_update_cluster<T, blocked>(data, cluster, center, norm_data, block_data, norm_center, block_center, Y, Z, min_dist);
		//std::cerr << "cluster changed " << changed2 << std::endl;
		if (!changed1 && !changed2) {
			std::cerr << "converges at step " << i << std::endl;
			break;
		}
		if (until_converge)
			max_interation += 1;
		nl = compute_loss(data, cluster, center);
		if (0 != i && nl - ll > 1) {
			std::cerr << "loss increase in step " << i << std::endl;
		}
		std::cerr << "step " << i << " loss " << nl << std::endl;
		ll = nl;
	}

	return 0;
}

#endif
