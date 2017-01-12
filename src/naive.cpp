#include "common.hpp"

#include <iostream>
#include <Eigen/Dense>
#include <memory>
#include <string>
#include <fstream>


bool naive_update_center(std::shared_ptr<KNN_config> conf, Eigen::MatrixXf &data, Eigen::VectorXi &cluster, Eigen::MatrixXf &center, Eigen::MatrixXd &workspace1, Eigen::VectorXi &workspace2) {
	bool changed = false;
	workspace1 = Eigen::MatrixXd::Zero(conf->data_dimension, conf->cluster_number);
	workspace2 = Eigen::VectorXi::Zero(conf->cluster_number);
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
	return changed;
}

bool naive_update_cluster(std::shared_ptr<KNN_config> conf, Eigen::MatrixXf &data, Eigen::VectorXi &cluster, Eigen::MatrixXf &center) {
	bool changed = false;
	for (int n = 0; n < conf->data_number; ++n) {
		int mp = 0;
		float mv = (data.col(n) - center.col(0)).squaredNorm();
		for (int k = 1; k < conf->cluster_number; ++k) {
			float nv = (data.col(n) - center.col(k)).squaredNorm();
			if (nv < mv)
				mp = k;
		}
		if (cluster(n) != mp)
			changed = true;
		cluster(n) = mp;
	}
	return changed;
}

int main(int argc, const char* argv[]) {
	std::shared_ptr<KNN_config> conf = KNN_parse_arg(argc, argv);
	if (!conf)
		return 0;
	Eigen::MatrixXf data(conf->data_dimension, conf->data_number);
	Eigen::VectorXf label(conf->data_number);
	Eigen::VectorXi cluster(conf->data_number);
	if (KNN_get_data(conf, data, label)) {
		return 0;
	}
	//std::cout << data << std::endl;
	if (conf->have_seed_file) {
		if (KNN_get_seed(conf, cluster))
			return 0;
		//std::cout << label << std::endl;
	}

	//generate_libsvm_data_file("test", conf, data, label);

	if (!conf->have_seed_file)
		generate_random_initial_cluster(conf, cluster);
	Eigen::MatrixXf center(conf->data_dimension, conf->cluster_number);
	Eigen::MatrixXd workspace1(conf->data_dimension, conf->cluster_number);
	Eigen::VectorXi workspace2(conf->cluster_number);
	for (int i = 0; i < conf->max_interation; ++i) {
		std::cerr << i << std::endl;
		bool changed = false;
		changed = changed || naive_update_center(conf, data, cluster, center, workspace1, workspace2);
		changed = changed || naive_update_cluster(conf, data, cluster, center);
		if (!changed) {
			std::cerr << "converges at step " << i << std::endl;
			break;
		}
		if (conf->until_converge)
			conf->max_interation += 1;
		std::cerr << "step " << i << " loss " << compute_loss(conf, data, cluster, center);
	}
	output_cluster(conf, cluster);

	return 0;
}

