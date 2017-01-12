#include "common.hpp"

#include <string>
#include <boost/program_options.hpp>
#include <boost/program_options/parsers.hpp>
#include <iostream>
#include <memory>

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

	return os;
}

std::unique_ptr<KNN_config> KNN_parse_arg(int argc, const char *argv[]) {
	std::unique_ptr<KNN_config> conf(new KNN_config);
	po::options_description desc("Allowed options");
	desc.add_options()
		("help,h", "produce help message")
		("data_file_name,f", po::value<std::string>(&conf->data_file_name), "libsvm format data file name")
		("seed_file_name,s", po::value<std::string>(&conf->seed_file_name), "initial KNN clustering")
		("data_number,n", po::value<int>(&conf->data_number), "number of data points")
		("data_dimension,d", po::value<int>(&conf->data_dimension), "dimension of data points")
		("cluster_number,k", po::value<int>(&conf->cluster_number), "number of clusters");
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
