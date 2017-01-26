#include "common.hpp"

#include <string>
#include <boost/program_options.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/random.hpp>
#include <ctime>
#include <cstdint>
#include <iostream>
#include <memory>
#include <cstdio>
#include <Eigen/Dense>
#include <fstream>
#include <iomanip>

std::ostream& operator<<(std::ostream& os, const KMeans_config& kc) {
	os << "data file name: " << kc.data_file_name << std::endl;
	os << "data number: " << kc.data_number << std::endl;
	os << "data dimension: " << kc.data_dimension << std::endl;
	os << "cluster number: " << kc.cluster_number << std::endl;
	if (kc.input_seed) {
		os << "have seed file: " << kc.input_seed_file_name << std::endl;
	} else {
		os << "don't have seed file" << std::endl;
	}
	os << "maximum iteration number: " << kc.max_interation << std::endl;
	os << "output file name: " << kc.output_file_name << std::endl;
	os << "precision for norm: " << kc.norm_precision << std::endl;
	if (kc.until_converge)
		os << "maximum iteration step unlimitted" << std::endl;
	else
		os << "maximum iteration step: " << kc.max_interation << std::endl;

	return os;
}

PConf KMeans_parse_arg(int argc, const char *argv[]) {
	namespace po = boost::program_options;

	std::shared_ptr<KMeans_config> conf(new KMeans_config);
	po::options_description desc("Allowed options");
	desc.add_options()
		("help,h", "produce help message")
		("data_file_name,f", po::value<std::string>(&conf->data_file_name), "libsvm format data file name")
		("output_file_name,o", po::value<std::string>(&conf->output_file_name)->default_value("clustering.output"), "clustering output file name")
		("input_seed_file_name,s", po::value<std::string>(&conf->input_seed_file_name), "file name for seed input")
		("output_seed_file_name,q", po::value<std::string>(&conf->output_seed_file_name), "file name for seed output")
		("data_number,n", po::value<int>(&conf->data_number), "number of data points")
		("data_dimension,d", po::value<int>(&conf->data_dimension), "dimension of data points")
		("cluster_number,k", po::value<int>(&conf->cluster_number), "number of clusters")
		("max_iteration,i", po::value<int>(&conf->max_interation)->default_value(-1), "maximum number of iteration")
		("group_number,g", po::value<int>(&conf->group_number), "number of center groups in yinyangkmeans, defaults to k / 10")
		("norm_precision,p", po::value<float>(&conf->norm_precision)->default_value(1e-4), "precision of the norm of the change of centers for judging convergenve")
		("yinyang,y", "yinyang kmeans")
		("kpp", "switch on kmeans++ initialization");
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);
	if (vm.count("help")) {
		std::cout << desc << "\n";
		return nullptr;
	}
	conf->kmeans_plus_plus_initialization = vm.count("kpp");
	conf->yinyang = vm.count("yinyang");
	conf->input_seed = vm.count("input_seed_file_name");
	conf->output_seed = vm.count("output_seed_file_name");
	if (-1 == conf->max_interation) {
		conf->max_interation = 2;
		conf->until_converge = true;
	}
	if (conf->yinyang) {
		if (!vm.count("group_number"))
			conf->group_number = conf->cluster_number / 10;
	}

	return conf;
}

int KMeans_get_data(std::shared_ptr<KMeans_config> conf, Eigen::MatrixXf &data, Eigen::VectorXf &label) {
	FILE* f = fopen(conf->data_file_name.c_str(), "r");
	int& N = conf->data_number;
	int& D = conf->data_dimension;
	bool err = false;
	for (int n = 0; n < N; ++n) {
		if (1 != fscanf(f, "%f%*[ ]", &label(n))) {
			err = true;
			break;
		}
		int c;
		while ('\n' != (c = getc(f))) {
			ungetc(c, f);
			int i;
			float v;
			if (2 != fscanf(f, "%d:%f%*[ ]", &i, &v) || i > D) {
				err = true;
				break;
			}
			data(i - 1, n) = v;
		}
	}
	fclose(f);
	if (err) {
		std::cerr << "Error in reading data file" << std::endl;
		return -1;
	}
	return 0;
}
/*
int KMeans_get_seed(std::shared_ptr<KMeans_config> conf, Eigen::VectorXi &cluster) {
	FILE *f = fopen(conf->input_seed_file_name.c_str(), "r");
	int& N = conf->data_number;
	int& K = conf->cluster_number;
	int n;
	fscanf(f, "%d", &n);
	if (n != N) {
		std::cerr << "Error in seed file: number of data wrong" << std::endl;
		fclose(f);
		return -1;
	}
	for (n = 0; n < N; ++n) {
		fscanf(f, "%d", &cluster(n));
		if (cluster(n) >= K) {
			std::cerr << "Error in seed file: value of cluster wrong" << std::endl;
			fclose(f);
			return -1;
		}
	}

	return 0;
}
*/
int generate_random_initial_cluster(PConf conf, DataMat &data, CenterMat &center) {
	boost::random::mt19937 gen{static_cast<std::uint32_t>(std::time(0))};
	boost::random::uniform_int_distribution<> dist{0, conf->cluster_number - 1};
	Eigen::VectorXi chosen(conf->data_number);
	chosen.setZero();
	for (int i = 0; i < conf->cluster_number; ++i) {
		int n;
		while (true) {
			n = dist(gen);
			if (!chosen(n)) {
				chosen(n) = 1;
				break;
			}
		}
		center.col(i) = data.col(n);
	}
	return 0;
}

int output_cluster(std::shared_ptr<KMeans_config>conf, Eigen::VectorXi &cluster) {
	std::ofstream fout(conf->output_file_name);
	fout << cluster << std::endl;
	fout.close();

	return 0;
}

double compute_loss(const DataMat &data, const ClusterVec &cluster, const CenterMat &center) {
	//std::cerr << "start compute loss" << std::endl;
	double l = 0;
	int N = data.cols();
	for (int n = 0; n < N; ++n) {
		/*
		std::cerr << "n " << n << std::endl;
		std::cerr << "cluster " << std::endl << cluster << std::endl;
		std::cerr << "cluster(n) " << cluster(n) << std::endl;
		*/
		l += (data.col(n).cast<double>() - center.col(cluster(n)).cast<double>()).norm();
	}
	//std::cerr << "finished compute loss" << std::endl;
	return l;
}

int generate_libsvm_data_file(std::string file_name, std::shared_ptr<KMeans_config> conf, Eigen::MatrixXf &data, Eigen::VectorXf &label) {
	std::ofstream fout(file_name);
	fout << std::fixed << std::setprecision(6);
	int &N = conf->data_number;
	int &K = conf->data_dimension;
	for (int n = 0; n < N; ++n) {
		fout << int(label(n)) << ' ';
		for (int k = 0; k < K; ++k)
			fout << k+1 << ":" << data(k, n) << ' ';
		fout << '\n';
	}
	fout.close();

	return 0;
}

int KMeans_export_seed(std::string &file_name, CenterMat &center) {
	std::ofstream fout(file_name.c_str(), std::ios::out|std::ios::binary);
	// warning: don't do this across machines
	int K = center.cols(), D = center.rows();
	fout.write((char*)&K, sizeof(int));
	fout.write((char*)&D, sizeof(int));
	for (int k = 0; k < K; ++k)
		for (int d = 0; d < D; ++d)
			fout.write((char*)&center(d, k), sizeof(double));
	fout.close();

	return 0;
}

int KMeans_load_seed(std::string &file_name, int &K, int &D, CenterMat &center) {
	std::ifstream fin(file_name.c_str(), std::ios::in|std::ios::binary);
	// warning: don't do this across machines
	fin.read((char*)&K, sizeof(int));
	fin.read((char*)&D, sizeof(int));
	std::cerr << "K: " << K << "D: " << D << std::endl;
	center.resize(D, K);
	for (int k = 0; k < K; ++k)
		for (int d = 0; d < D; ++d)
			fin.read((char*)&center(d, k), sizeof(double));
	fin.close();

	return 0;
}
