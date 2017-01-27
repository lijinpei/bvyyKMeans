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
		("debug", "switch on this option to compare yinyang kmeans with lloyd kmeans")
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
	conf->debug = vm.count("debug");
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

int output_cluster(std::shared_ptr<KMeans_config>conf, ClusterVec &cluster) {
	std::ofstream fout(conf->output_file_name);
	for (auto i: cluster)
		fout << i << ' ';
	fout.close();

	return 0;
}

