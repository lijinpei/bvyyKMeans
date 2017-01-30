#include "common.hpp"
#include "lloyd.hpp"
#include "yinyang.hpp"
#include "kmeans_plus_plus.hpp"
#include <iostream>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_sparse.hpp>
#include <boost/numeric/ublas/io.hpp>

template <class T>
int run_main(PConf conf) {
	int N = conf->data_number;
	int K = conf->cluster_number;
	int D = conf->data_dimension;
	DataMat<T> data(N, T(D));
	LabelVec label(N);
	ClusterVec cluster(N);
	CenterMat<T> center(K, T(D));
	if (KMeans_get_data(conf, data, label)) {
		std::cerr << "error when get data" << std::endl;
		return 1;
	}
/*
	std::cout << conf->input_seed << std::endl;
	std::cout << conf->kmeans_plus_plus_initialization << std::endl;
	std::cout << conf->block_size << std::endl;
*/
	if (conf->input_seed) {
		int tmp_K = -1, tmp_D = -1;
		std::cerr << "start load seed" << std::endl;
		if (KMeans_load_seed(conf->input_seed_file_name, tmp_K, tmp_D, center)) {
			std::cerr << "error when get seed" << std::endl;
			return 0;
		}
		if (K != tmp_K)
			std::cerr << "error: wrong data number in seed file, get " << tmp_K << " expect " << K << std::endl;
		if (D != tmp_D)
			std::cerr << "error: wrong data dimension in seed file, get" << tmp_D << " expect " << D << std::endl;
		if (K!= tmp_K || D != tmp_D)
			return 0;
	} else if (conf->kmeans_plus_plus_initialization) {
		if (kmeans_plus_plus_initialize(data, center)) {
			std::cerr << "error when kmeans plus plus initialization" << std::endl;
		}
	} else
		generate_random_initial_cluster(conf, data, center);
	if (conf->output_seed)
		KMeans_export_seed(conf->output_seed_file_name, center, K, D);

	const int B = conf->block_size;
	std::vector<double> norm_data, norm_center;
	std::vector<T> block_data, block_center;
	if (B > 0) {
		norm_data.resize(N);
		block_data = std::vector<T>(N, T(B));
		for (int n = 0; n < N; ++n) {
			norm_data[n] = bvyyKMeansNorm(data[n]);
			generate_block_vector(block_data[n], data[n], B, D);
		}
		norm_center.resize(K);
		block_center = std::vector<T>(K, T(B));
		for (int k = 0; k < K; ++k) {
			norm_center[k] = bvyyKMeansNorm(center[k]);
			generate_block_vector(block_center[k], center[k], B, D);
		}
	}
	if (conf->yinyang) {
		int G = conf->group_number;
		if (B > 0)
			yinyang<T, true>(data, cluster, center, D, G, B, conf->norm_precision, conf->max_interation, conf->until_converge, conf->debug, norm_data, block_data, norm_center, block_center);
		else
			yinyang<T, false>(data, cluster, center, D, G, B, conf->norm_precision, conf->max_interation, conf->until_converge, conf->debug, norm_data, block_data, norm_center, block_center);
	} else {
		if (B > 0)
			lloyd<T, true>(data, cluster, center, conf->norm_precision, D, conf->max_interation, conf->until_converge, B, norm_data, block_data, norm_center, block_center);
		else
			lloyd<T, false>(data, cluster, center, conf->norm_precision, D, conf->max_interation, conf->until_converge, B, norm_data, block_data, norm_center, block_center);
	}
	output_cluster(conf, cluster);

	return 0;
}

int main(int argc, const char* argv[]) {
	std::shared_ptr<KMeans_config> conf = KMeans_parse_arg(argc, argv);
	if (!conf)
		return 0;
	if (conf->sparse) 
		return run_main<SparseVec<float>>(conf);
	else
		return run_main<DenseVec<float>>(conf);
}

