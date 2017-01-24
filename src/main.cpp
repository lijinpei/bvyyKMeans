#include "common.hpp"
#include "lloyd.hpp"
#include "kmeans_plus_plus.hpp"
#include <iostream>


int main(int argc, const char* argv[]) {
	std::shared_ptr<KMEANS_config> conf = KMEANS_parse_arg(argc, argv);
	if (!conf)
		return 0;
	int N = conf->data_number;
	int K = conf->cluster_number;
	int D = conf->data_dimension;
	DataMat data(D, N);
	LabelVec label(N);
	ClusterVec cluster(N);
	CenterMat center(D, K);
	if (KMEANS_get_data(conf, data, label)) {
		std::cerr << "error when get data" << std::endl;
		return 1;
	}

	if (conf->input_seed) {
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
	
	if (conf->yinyang) {
		yinyang(data, cluster, center, conf->norm_precision, conf->max_interation, conf->until_converge);
	} else {
		lloyd(data, cluster, center, conf->norm_precision, conf->max_interation, conf->until_converge); 
	}
	output_cluster(conf, cluster);

	return 0;
}

