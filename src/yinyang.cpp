#include "common.hpp"
#include "lloyd.hpp"
#include "kmeans_plus_plus.hpp"
#include <Eigen/Dense>

Eigen::VectorXi group;		// which group a cluster center is in
Eigen::MatrixXf lbg;		// lower bound for group
Eigen::VectorXf delta_c;	// delta change of cluster center
Eigen::VectorXf delta_g;	// max delta change of cluster group
Eigen::MatrixXf center_sum;	// sum of all points in certain cluster
Eigen::VectorXf ub;		// upper bound for d(x, b(x))

// first iteration of yinyangkmeans
// ran when cluster center are just generated
// divide center into groups and initialize other values
int yinyang_first_iteration(int G, DataMat &data, CenterMat &center, ClusterVec &cluster, double precision) {
	int D = data.rows();
	CenterMat center_center(D, G);
	group.resize(G);
	kmeans_plus_plus_initialize(center, center_center);
	lloyd(center, group, center_center, precision, 5, false);
	int N = data.cols();
	int K = center.cols();
	Eigen::MatrixXd d(K, N);
	for (int n = 0; n < N; ++n)
		for (int k = 0; k < K; ++k)
			d(k, n) = (center.col(k) - data.col(n)).norm();
	ub.resize(N);
	for (int n = 0; n < N; ++n)
		ub(n) = d.col(n).minCoeff(&cluster(n));
	lbg.resize(G, N);
	lbg = decltype(lbg)::Constant(lbg.rows(), lbg.cols(), -1);
	center_sum.resize(D, K);
	for (int n = 0; n < N; ++n)
		for (int k = 0; k < K; ++k)
			if (k != cluster(n)) {
				int g = group(k);
				double tmp_d = d(k, n);
				float tmp_lbg = lbg(g, n);
				if (tmp_lbg < 0 || tmp_d < tmp_lbg)
					lbg(g, n) = tmp_d;
			} else
				center_sum.col(k) += data.col(n);
	return 0;
}

int yinyang(const DataMat &data, ClusterVec &cluster, CenterMat &center, double precision, int max_interation, bool until_converge) {
	return 0;
}
