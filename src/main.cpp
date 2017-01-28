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
		KMeans_export_seed(conf->output_seed_file_name, center);

	if (conf->yinyang) {
		int G = conf->group_number;
		yinyang(data, cluster, center, D, G, conf->norm_precision, conf->max_interation, conf->until_converge, conf->debug);
	} else {
		lloyd(data, cluster, center, conf->norm_precision, conf->max_interation, conf->until_converge); 
	}
	output_cluster(conf, cluster);

	return 0;
}

int main(int argc, const char* argv[]) {
	std::shared_ptr<KMeans_config> conf = KMeans_parse_arg(argc, argv);
	if (!conf)
		return 0;
	if (conf->sparse) 
		return run_main<boost::numeric::ublas::compressed_vector<float>>(conf);
	else
		return run_main<boost::numeric::ublas::vector<float>>(conf);
}

