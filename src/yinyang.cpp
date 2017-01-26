#include "common.hpp"
#include "lloyd.hpp"
#include "kmeans_plus_plus.hpp"
#include <Eigen/Dense>
#include <type_traits>
#include <vector>
#include <set>
#include <iostream>


// first iteration of yinyangkmeans
// ran when cluster center are just generated
// divide center into groups and initialize other values
int yinyang_first_iteration(const DataMat &data, ClusterVec &cluster, CenterMat &center, int G, double precision, Eigen::VectorXi &group, Eigen::MatrixXf &lbg, Eigen::VectorXf &ub, Eigen::MatrixXf &center_sum, Eigen::VectorXi &center_count) {
	std::cerr << "start yinyang first iteration" << std::endl;
	int D = data.rows();
	int N = data.cols();
	int K = center.cols();
	CenterMat center_center(D, G);
	kmeans_plus_plus_initialize(center, center_center);
	lloyd(center, group, center_center, precision, 5, false);
	/* run lloyd kmeans for the first iteration */
	Eigen::MatrixXd workspace1(D, K);
	Eigen::VectorXi workspace2(K);
	lloyd_update_cluster(data, cluster, center);
	lloyd_update_center(data, cluster, center, precision, workspace1, workspace2);
	Eigen::MatrixXd d(K, N);
	for (int n = 0; n < N; ++n)
		for (int k = 0; k < K; ++k)
			d(k, n) = (center.col(k) - data.col(n)).norm();
	/* initialize ub, lbg */
	for (int n = 0; n < N; ++n) {
		//std::cerr << "d.col(n) " << d.col(n) << std::endl;
		ub(n) = d.col(n).minCoeff(&cluster(n));
	}
	//lbg = std::remove_reference<decltype(lbg)>::type::Constant(G, N, -1);
	lbg.setConstant(-1);
	center_count.setZero();
	for (int n = 0; n < N; ++n)
		for (int k = 0; k < K; ++k)
			if (k != cluster(n)) {
				int g = group(k);
				double tmp_d = d(k, n);
				float tmp_lbg = lbg(g, n);
				if (tmp_lbg < 0 || tmp_d < tmp_lbg)
					lbg(g, n) = tmp_d;
			} else {
				center_count(k) += 1;
				center_sum.col(k) += data.col(n);
			}
	std::cerr << "finished yinyang first iteration" << std::endl;
	return 0;
}

bool update_center(CenterMat &center, Eigen::VectorXi &group, Eigen::MatrixXf &center_sum, Eigen::VectorXi &center_count, Eigen::VectorXf &delta_c, Eigen::VectorXf &delta_g, double precision) {
	std::cerr << "start update_center" << std::endl;
	bool changed = false;
	Eigen::VectorXf tmp_center;
	int K = center.cols();
	delta_g.setConstant(-1);
	for (int k = 0; k < K; ++k) {
		int cc = center_count(k);
		if (!cc)
			continue;
		tmp_center = center.col(k);
		center.col(k) = center_sum.col(k) / cc;
		float dc = (center.col(k) - tmp_center).norm();
		if (dc > precision)
			changed = true;
		delta_c(k) = dc;
		int g = group(k);
		float dg = delta_g(g);
		if (dg < dc)
			delta_g(g) = dc;
	}
	std::cerr << "finished update_center" << std::endl;
	return changed;
}

bool yinyang_update_cluster(const DataMat &data, ClusterVec &cluster, CenterMat &center, Eigen::VectorXi &group, std::vector<std::set<int>> &centers_in_group, Eigen::MatrixXf &lbg, Eigen::VectorXf &ub, Eigen::VectorXf &delta_c, Eigen::VectorXf &delta_g, Eigen::MatrixXf &center_sum, Eigen::VectorXi &center_count) {
	std::cerr << "start yinyang_update_cluster" << std::endl;
	bool changed = false;
	int N = data.cols();
	int G = lbg.rows();
	for (int n = 0; n < N; ++n) {
		ub(n) += delta_c(cluster(n));
		Eigen::VectorXf old_lbg(lbg.col(n));
		lbg.col(n) -= delta_g;
		/* Global filtering */
		float min_lbg = -1;
		for (int g = 0; g < G; ++g) {
			double tmp_lbg = lbg(g, n);
			if (tmp_lbg < 0)
				continue;
			if (min_lbg < 0 || min_lbg > tmp_lbg)
				min_lbg = tmp_lbg;
		}
		if (min_lbg > ub(n))
			continue;
		ub(n) = (data.col(n) - center.col(cluster(n))).norm();
		if (min_lbg > ub(n))
			continue;
		/* Group filtering */
		for (int g = 0; g < G; ++g) {
			if (lbg(g, n) > ub(n))
				continue;
			/* Local filtering */
			for (int c:centers_in_group[g]) {
				if (c == cluster(n))
					continue;
				if (old_lbg(g) - delta_c(c) > ub(n))
					continue;
				float tmp_d = (data.col(n) - center.col(c)).norm();
				if (tmp_d < ub(n)) {
					changed = true;
					int l = cluster(n);
					lbg(group(l), n) = ub(n);
					cluster(n) = c;
					ub(n) = tmp_d;
					center_count(l) -= 1;
					center_count(c) += 1;
					center_sum.col(l) -= data.col(n);
					center_sum.col(c) += data.col(n);
				} else {
					if (tmp_d < lbg(g, n))
						lbg(g, n) = tmp_d;
				}
			}
		}
	}
	std::cerr << "finished yinyang_update_cluster" << std::endl;
	return changed;
}

int yinyang(const DataMat &data, ClusterVec &cluster, CenterMat &center, int G, double precision, int max_iteration, bool until_converge) {
	int N = data.cols();
	int D = data.rows();
	int K = center.cols();

	std::vector< std::set<int> > centers_in_group(G);
	Eigen::VectorXi group(K);		// which group a cluster center is in
	Eigen::MatrixXf lbg(G, N);		// lower bound for group
	Eigen::VectorXf delta_c(K);		// delta change of cluster center
	Eigen::VectorXf delta_g(G);		// max delta change of cluster group
	Eigen::VectorXf ub(N);			// upper bound for d(x, b(x))
	Eigen::MatrixXf center_sum(D, K);	// sum of all points in certain cluster
	Eigen::VectorXi center_count(K);	// how many points in a cluster

	yinyang_first_iteration(data, cluster, center, G, precision, group, lbg, ub, center_sum, center_count);
	//std::cerr << cluster;
	std::cerr << "max iteration " << max_iteration << std::endl;
	double ll = compute_loss(data, cluster, center), nl;
	for (int it = 1; it < max_iteration; ++it) {
		bool changed = false;
		changed = changed || update_center(center, group, center_sum, center_count, delta_c, delta_g, precision);
		changed = changed || yinyang_update_cluster(data, cluster, center, group, centers_in_group, lbg, ub, delta_c, delta_g, center_sum, center_count);
		if (!changed) {
			std::cerr << "converges at step " << it << std::endl;
			break;
		}
		if (until_converge)
			max_iteration += 1;
		nl = compute_loss(data, cluster, center);
		if (nl - ll > 1) {
			std::cerr << "loss increase in step " << it << std::endl;
		}
		std::cerr << "step " << it << " loss " << nl << std::endl;
		ll = nl;
		if (until_converge)
			++max_iteration;
		if (!changed)
			break;
	}

	return 0;
}
