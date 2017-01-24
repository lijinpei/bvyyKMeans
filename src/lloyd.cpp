#include "common.hpp"
#include <iostream>
#include <Eigen/Dense>
#include <memory>


bool naive_update_center(DataMat &data, ClusterVec &cluster, CenterMat &center, double precision, Eigen::MatrixXd &workspace1, Eigen::VectorXi &workspace2) {
	int N = data.cols();
	int K = center.cols();
	double l1 = compute_loss(data, cluster, center);
	bool changed = false;
	workspace1.setZero();
	workspace2.setZero();
	for (int n = 0; n < N; ++n) {
		int c = cluster(n);
		workspace2(c) += 1;
		workspace1.col(c) += data.col(n).cast<double>();
	}
	for (int k = 0; k < K; ++k)
		if (0 == workspace2(k)) {
			std::cerr << "zero data point in cluster" << std::endl;
			continue;
		} else {
			Eigen::VectorXf new_center = (workspace1.col(k) / workspace2(k)).cast<float>();
			if ((new_center - center.col(k)).norm() > precision)
				changed = true;
			center.col(k) = new_center;
		}
	double l2 = compute_loss(data, cluster, center);
	if (l2 - l1 > 1)
		std::cerr << "Loss increases in update center" << std::endl;
	return changed;
}

bool naive_update_cluster(DataMat &data, ClusterVec &cluster, CenterMat &center) {
	double l1 = compute_loss(data, cluster, center);
	bool changed = false;
	int N = data.cols();
	int K = center.cols();
	for (int n = 0; n < N; ++n) {
		int mp = 0;
		float mv = (data.col(n) - center.col(0)).squaredNorm();
		for (int k = 1; k < K; ++k) {
			float nv = (data.col(n) - center.col(k)).squaredNorm();
			if (nv < mv) {
				mp = k;
				mv = nv;
			}
		}
		if (cluster(n) != mp)
			changed = true;
		double d1 = (data.col(n) - center.col(mp)).squaredNorm();
		double d2 = (data.col(n) - center.col(cluster(n))).squaredNorm();
		if (mp != cluster(n) && d1 > d2)
			std::cerr << "Loss increases in single update cluster " << mp << " " << cluster(n) << std::endl;
		cluster(n) = mp;
	}
	double l2 = compute_loss(data, cluster, center);
	if (l2 > l1)
		std::cerr << "Loss increases in update cluster" << std::endl;
	return changed;
}

int lloyd(const DataMat &data, ClusterVec &cluster, CenterMat &center, double precision, int max_interation, bool until_converge) {
	int K = center.cols();
	int D = data.cols();
	if (D != center.cols()) {
		return 1;
	}
	Eigen::MatrixXd workspace1(D, K);
	Eigen::VectorXi workspace2(K);
	double ll;
	double nl;
	for (int i = 0; i < max_interation; ++i) {
		bool changed = false;
		changed = changed || naive_update_cluster(data, cluster, center);
		changed = changed || naive_update_center(data, cluster, center, precision, workspace1, workspace2);
		if (!changed) {
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
}

