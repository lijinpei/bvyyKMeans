#include "common.hpp"
#include "lloyd.hpp"
#include "kmeans_plus_plus.hpp"
#include <Eigen/Dense>
#include <type_traits>
#include <vector>
#include <set>
#include <iostream>
#include <cmath>


// first iteration of yinyangkmeans
// ran when cluster center are just generated
// divide center into groups and initialize other values
int yinyang_first_iteration(const DataMat &data, ClusterVec &cluster, CenterMat &center, int G, double precision, Eigen::VectorXi &group, Eigen::MatrixXf &lbg, Eigen::VectorXf &ub, Eigen::MatrixXf &center_sum, Eigen::VectorXi &center_count, std::vector<std::set<int>> &centers_in_group) {
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
	lloyd_update_cluster(data, cluster, center);
	std::cerr << "step 0 loss " << compute_loss(data, cluster, center) << std::endl;
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
	center_sum.setZero();
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
	centers_in_group.resize(G);
	for (int k = 0; k < K; ++k)
		centers_in_group[group[k]].insert(k);
	std::cerr << "finished yinyang first iteration" << std::endl;
	return 0;
}

bool update_center(CenterMat &center, Eigen::VectorXi &group, Eigen::MatrixXf &center_sum, Eigen::VectorXi &center_count, Eigen::VectorXf &delta_c, Eigen::VectorXf &delta_g, double precision) {
	//std::cerr << "start update_center" << std::endl;
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
	//std::cerr << "finished update_center" << std::endl;
	return changed;
}

bool yinyang_update_cluster(const DataMat &data, ClusterVec &cluster, CenterMat &center, Eigen::VectorXi &group, std::vector<std::set<int>> &centers_in_group, Eigen::MatrixXf &lbg, Eigen::VectorXf &ub, Eigen::VectorXf &delta_c, Eigen::VectorXf &delta_g, Eigen::MatrixXf &center_sum, Eigen::VectorXi &center_count) {
	//std::cerr << "start yinyang_update_cluster" << std::endl;
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
		//std::cerr << "pass global filtering ";
		/* Group filtering */
		for (int g = 0; g < G; ++g) {
			if (lbg(g, n) > ub(n))
				continue;
			//std::cerr << "pass group filtering ";
			/* Local filtering */
			for (int c:centers_in_group[g]) {
				//std::cerr << "start local filtering ";
				if (c == cluster(n))
					continue;
				if (old_lbg(g) - delta_c(c) > ub(n))
					continue;
				float tmp_d = (data.col(n) - center.col(c)).norm();
				if (tmp_d < ub(n)) {
					//std::cerr << "pass local filtering ";
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
	//std::cerr << "finished yinyang_update_cluster" << std::endl;
	return changed;
}

int cmp_center(CenterMat &center, CenterMat &center1, double precision) {
	int K = center.cols();
	int D = center.rows();
	for (int k = 0; k < K; ++k) {
		for (int d = 0; d < D; ++d)
			if (std::abs(center(d, k) - center1(d, k)) > precision) {
				return 1;
			}
	}

	return 0;
}

int cmp_cluster(ClusterVec &cluster, ClusterVec &cluster1) {
	int N = cluster.rows();
	for (int n = 0; n < N; ++n)
		if (cluster(n) != cluster1(n))
			return 1;
	return 0;
}

int yinyang(const DataMat &data, ClusterVec &cluster, CenterMat &center, int G, double precision, int max_iteration, bool until_converge, bool debug) {
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

	ClusterVec cluster1;
	CenterMat center1;
	Eigen::MatrixXd workspace1;
	Eigen::VectorXi workspace2;
	if (debug) {
		cluster1.resize(N);
		center1.resize(D, K);
		workspace1.resize(D, N);
		workspace2.resize(K);
	}
	yinyang_first_iteration(data, cluster, center, G, precision, group, lbg, ub, center_sum, center_count, centers_in_group);
	//std::cerr << cluster;
	//std::cerr << "max iteration " << max_iteration << std::endl;
	double ll = compute_loss(data, cluster, center), nl;
	for (int it = 1; it < max_iteration; ++it) {
		bool changed1, changed2;
		changed1 = update_center(center, group, center_sum, center_count, delta_c, delta_g, precision);
		//std::cerr << "center changed " << changed1 << std::endl;
		nl = compute_loss(data, cluster, center);
		if (nl - ll > 1) {
			std::cerr << "loss increase in update center in step " << it << std::endl;
		}
		ll = nl;
		if (debug) {
			lloyd_update_center(data, cluster, center1, precision, workspace1, workspace2);
			if (cmp_center(center, center1, precision)) {
				std::cerr << "different center in step " << it << std::endl;
			}
		}
		changed2 = yinyang_update_cluster(data, cluster, center, group, centers_in_group, lbg, ub, delta_c, delta_g, center_sum, center_count);
		//std::cerr << "cluster changed " << changed2 << std::endl;
		nl = compute_loss(data, cluster, center);
		if (nl - ll > 1) {
			std::cerr << "loss increase in update cluster in step " << it << std::endl;
		}
		ll = nl;
		if (debug) {
			lloyd_update_cluster(data, cluster1, center);
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
		std::cerr << "step " << it << " loss " << nl << std::endl;
		ll = nl;
		if (until_converge)
			++max_iteration;
	}

	return 0;
}
