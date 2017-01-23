#include "common.hpp"

#include <iostream>
#include <Eigen/Dense>
#include <memory>
#include <string>
#include <fstream>


bool naive_update_center(std::shared_ptr<KMEANS_config> conf, Eigen::MatrixXf &data, Eigen::VectorXi &cluster, Eigen::MatrixXf &center, Eigen::MatrixXd &workspace1, Eigen::VectorXi &workspace2) {
	double l1 = compute_loss(conf, data, cluster, center);
	bool changed = false;
	workspace1.setZero();
	workspace2.setZero();
	for (int n = 0; n < conf->data_number; ++n) {
		int c = cluster(n);
		workspace2(c) += 1;
		workspace1.col(c) += data.col(n).cast<double>();
	}
	for (int k = 0; k < conf->cluster_number; ++k)
		if (0 == workspace2(k)) {
			std::cerr << "zero data point in cluster" << std::endl;
			continue;
		} else {
			Eigen::VectorXf new_center = (workspace1.col(k) / workspace2(k)).cast<float>();
			if ((new_center - center.col(k)).norm() > conf->norm_precision)
				changed = true;
			center.col(k) = new_center;
		}
	double l2 = compute_loss(conf, data, cluster, center);
	if (l2 - l1 > 1)
		std::cerr << "Loss increases in update center" << std::endl;
	return changed;
}

bool naive_update_cluster(std::shared_ptr<KMEANS_config> conf, Eigen::MatrixXf &data, Eigen::VectorXi &cluster, Eigen::MatrixXf &center) {
	double l1 = compute_loss(conf, data, cluster, center);
	bool changed = false;
	for (int n = 0; n < conf->data_number; ++n) {
		int mp = 0;
		float mv = (data.col(n) - center.col(0)).squaredNorm();
		for (int k = 1; k < conf->cluster_number; ++k) {
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
	double l2 = compute_loss(conf, data, cluster, center);
	if (l2 > l1)
		std::cerr << "Loss increases in update cluster" << std::endl;
	return changed;
}

int main(int argc, const char* argv[]) {
	std::shared_ptr<KMEANS_config> conf = KMEANS_parse_arg(argc, argv);
	if (!conf)
		return 0;
	DataMat data(conf->data_dimension, conf->data_number);
	LabelVec label(conf->data_number);
	ClusterVec cluster(conf->data_number);
	CenterMat center(conf->data_dimension, conf->cluster_number);
	if (KMEANS_get_data(conf, data, label)) {
		std::cerr << "error when get data" << std::endl;
		return 1;
	}

	if (conf->have_seed_file) {
		if (KMEANS_get_seed(conf, cluster)) {
			std::cerr << "error when get seed" << std::endl;
			return 0;
		}
	} else if (conf->kmeans_plus_plus_initialization) {
		if (kmeans_plus_plus_initialize(conf, data, center)) {
			std::cerr << "error when kmeans plus plus initialization" << std::endl;
		}
	} else
		generate_random_initial_cluster(conf, data, center);

	//generate_libsvm_data_file("test", conf, data, label);

	Eigen::MatrixXd workspace1(conf->data_dimension, conf->cluster_number);
	Eigen::VectorXi workspace2(conf->cluster_number);
	double ll;
	double nl;
	for (int i = 0; i < conf->max_interation; ++i) {
		bool changed = false;
		changed = changed || naive_update_cluster(conf, data, cluster, center);
		changed = changed || naive_update_center(conf, data, cluster, center, workspace1, workspace2);
		if (!changed) {
			std::cerr << "converges at step " << i << std::endl;
			break;
		}
		if (conf->until_converge)
			conf->max_interation += 1;
		nl = compute_loss(conf, data, cluster, center);
		if (0 != i && nl - ll > 1) {
			std::cerr << "loss increase in step " << i << std::endl;
		}
		std::cerr << "step " << i << " loss " << nl << std::endl;
		ll = nl;
	}
	output_cluster(conf, cluster);

	return 0;
}

