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

namespace po = boost::program_options;

std::ostream& operator<<(std::ostream& os, const KNN_config& kc) {
	os << "data file name: " << kc.data_file_name << std::endl;
	os << "data number: " << kc.data_number << std::endl;
	os << "data dimension: " << kc.data_dimension << std::endl;
	os << "cluster number: " << kc.cluster_number << std::endl;
	if (kc.have_seed_file) {
		os << "have seed file: " << kc.seed_file_name << std::endl;
	} else {
		os << "don't have seed file" << std::endl;
	}
	os << "maximum iteration number: " << kc.max_interation << std::endl;
	os << "output file name: " << kc.output_file_name << std::endl;
	os << "precision for norm: " << kc.norm_precision << std::endl;

	return os;
}

std::shared_ptr<KNN_config> KNN_parse_arg(int argc, const char *argv[]) {
	std::shared_ptr<KNN_config> conf(new KNN_config);
	po::options_description desc("Allowed options");
	desc.add_options()
		("help,h", "produce help message")
		("data_file_name,f", po::value<std::string>(&conf->data_file_name), "libsvm format data file name")
		("output_file_name,o", po::value<std::string>(&conf->output_file_name)->default_value("clustering.output"), "clustering output file name")
		("seed_file_name,s", po::value<std::string>(&conf->seed_file_name), "initial KNN clustering")
		("data_number,n", po::value<int>(&conf->data_number), "number of data points")
		("data_dimension,d", po::value<int>(&conf->data_dimension), "dimension of data points")
		("cluster_number,k", po::value<int>(&conf->cluster_number), "number of clusters")
		("max_iteration,i", po::value<int>(&conf->max_interation)->default_value(-1), "maximum number of iteration")
		("norm_precision,p", po::value<float>(&conf->norm_precision)->default_value(1e-4), "precision of the norm of the change of centers for judging convergenve");
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);
	conf->have_seed_file = vm.count("seed_file_name");
	if (vm.count("help")) {
		std::cout << desc << "\n";
		return nullptr;
	}

	return conf;
}

int KNN_get_data(std::shared_ptr<KNN_config> conf, Eigen::MatrixXf &data, Eigen::VectorXf &label) {
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

int KNN_get_seed(std::shared_ptr<KNN_config> conf, Eigen::VectorXi &cluster) {
	FILE *f = fopen(conf->seed_file_name.c_str(), "r");
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

int generate_random_initial_cluster(std::shared_ptr<KNN_config> conf, Eigen::VectorXi &cluster) {
	std::time_t now = std::time(0);
	boost::random::mt19937 gen{static_cast<std::uint32_t>(now)};
	boost::random::uniform_int_distribution<> dist{0, conf->cluster_number - 1};
	for (int n = 0; n < conf->data_number; ++n)
		cluster(n) = dist(gen);
	return 0;
}

void output_cluster(std::shared_ptr<KNN_config>conf, Eigen::VectorXi cluster) {
	std::ofstream fout(conf->output_file_name);
	fout << cluster << std::endl;
	fout.close();
}

void generate_libsvm_data_file(std::string file_name, std::shared_ptr<KNN_config> conf, Eigen::MatrixXf &data, Eigen::VectorXf &label) {
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

}
