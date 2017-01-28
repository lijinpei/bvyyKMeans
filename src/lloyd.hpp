#ifndef LLOYD_HPP
#define LLOYD_HPP
#include "common.hpp"

template <class T>
bool lloyd_update_center(const DataMat<T> &data, const ClusterVec &cluster, CenterMat<T> &center, double precision, CenterMat<T> &workspace1, ClusterVec &workspace2) {
	//std::cerr << "start lloyd update center" << std::endl;
	const int N = data.size();
	const int K = center.size();
	double l1 = compute_loss(data, cluster, center);
	bool changed = false;
	for (int k = 0; k < K; ++k)
		workspace1[k].clear();
	std::fill(workspace2.begin(), workspace2.end(), 0);
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
			center[k] = tmp_vec;
		}
	double l2 = compute_loss(data, cluster, center);
	if (l2 - l1 > 1)
		std::cerr << "Loss increases in update center" << std::endl;
	//std::cerr << "finished lloyd update center" << std::endl;
	return changed;
}

template <class T>
bool lloyd_update_cluster(const DataMat<T> &data, ClusterVec &cluster, const CenterMat<T> &center) {
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
			float nv = bvyyKMeansDistance(data[n], center[k]);
			if (nv < mv) {
				mp = k;
				mv = nv;
			}
		}
		if (cluster[n] != mp)
			changed = true;
		double d1 = bvyyKMeansDistance(data[n], center[mp]);
		double d2 = bvyyKMeansDistance(data[n], center[cluster[n]]);
		if (cluster[n] != mp && d1 > d2)
			std::cerr << "Loss increases in single update cluster " << mp << " " << cluster[n] << std::endl;
		cluster[n] = mp;
	}
	double l2 = compute_loss(data, cluster, center);
	if (l2 > l1)
		std::cerr << "Loss increases in update cluster" << std::endl;
	//std::cerr << "finished lloyd update cluster" << std::endl;
	return changed;
}
template <class T>
int lloyd(const DataMat<T> &data, ClusterVec &cluster, CenterMat<T> &center, double precision, int D, int max_interation, bool until_converge) {
	std::cerr << "start lloyd iteration" << std::endl;
	const int K = center.size();

	CenterMat<T> workspace1(K, T(D));
	ClusterVec workspace2(K);
	double ll;
	double nl;
	lloyd_update_cluster(data, cluster, center);
	for (int i = 0; i < max_interation; ++i) {
		//std::cerr << "start iteration " << i << std::endl;
		bool changed1, changed2;
		changed1 = lloyd_update_center(data, cluster, center, precision, workspace1, workspace2);
		//std::cerr << "center changed " << changed1 << std::endl;
		changed2 = lloyd_update_cluster(data, cluster, center);
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
