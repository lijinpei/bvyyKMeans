#include <string>
#include <boost/program_options.hpp>
#include <boost/program_options/parsers.hpp>
#include <iostream>

namespace po = boost::program_options;

int main(int argc, const char *argv[]) {
	std::string file_name;
	int data_number;
	int data_dimension;
	int cluster_number;
	po::options_description desc("Allowed options");
	desc.add_options()
		("help,h", "produce help message")
		("file_name,f", po::value<std::string>(&file_name), "libsvm format data file name")
		("data_number,n", po::value<int>(&data_number), "number of data points")
		("data_dimension,d", po::value<int>(&data_dimension), "dimension of data points")
		("cluster_number,k", po::value<int>(&cluster_number), "number of clusters");
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if (vm.count("help")) {
		std::cout << desc << "\n";
		return 0;
	}


	std::cout << vm["file_name"].as<std::string>() << std::endl;
	std::cout << vm["data_number"].as<int>() << std::endl;
	std::cout << vm["data_dimension"].as<int>() << std::endl;
	std::cout << vm["cluster_number"].as<int>() << std::endl;
	std::cout << std::endl;

	std::cout << file_name << std::endl;
	std::cout << data_number << std::endl;
	std::cout << data_dimension << std::endl;
	std::cout << cluster_number << std::endl;

	return 0;
}

