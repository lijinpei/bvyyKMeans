#include "common.hpp"
#include "preprocessing.hpp"
#include <boost/program_options.hpp>
#include <boost/program_options/parsers.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>

int main(int argc, char* argv[]) {
	namespace po = boost::program_options;
	std::vector<std::string> input_file_names;
	std::string output_file_name;
	bool input_libsvm_format;
	bool output_libsvm_format;
	bool need_scale;
	bool have_label;
	int tmp_target_min, tmp_target_max;

	int D;
	po::options_description desc("Allowed options");
	desc.add_options()
		("help,h", "produce help message")
		("input_file,i", po::value<std::vector<std::string>>(&input_file_names)->multitoken(), "name of input files")
		("output_file,o", po::value<std::string>(&output_file_name), "output file")
		("input_libsvm,v", "switch on this option to read input data in libsvm format")
		("output_libsvm,m", "switch on this option to write output data in libsvm format")
		("scale,s", "switch on this option to scale data")
		("label,l", "switch on this option to process data file with label")
		("dimension,d", po::value<int>(&D), "dimension of input data")
		("target_min,q", po::value<int>(&tmp_target_min)->default_value(-1), "min value of scaled output")
		("target_max,p", po::value<int>(&tmp_target_max)->default_value(1), "max value of scaled output");
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);
	if (vm.count("help")) {
		std::cout << desc << "\n";
		return 0;
	}
	bool option_error = false;
	if (!vm.count("input_file") || !vm.count("output_file")) {
		std::cout << "Must provide input and output file" << std::endl;
		option_error = true;
	}
	if (!vm.count("dimension")) {
		std::cout << "Must provide data dimension" << std::endl;
		option_error = true;
	}
	if (option_error) {
		std::cout << "Use " << argv[0] << " -h" << " to get detailed messages" << std::endl;
		return 0;
	}
	input_libsvm_format = vm.count("input_libsvm");
	output_libsvm_format = vm.count("output_libsvm");
	need_scale = vm.count("scale");
	have_label = vm.count("label");


	std::vector<std::vector<double>> data;
	std::vector<double> label;
	for (auto file_name: input_file_names) {
		std::cerr << file_name << std::endl;
		KMeans_read_data(file_name, data, label, D, have_label, input_libsvm_format);
	}
	std::cerr << "finished read input filles" << std::endl;
	if (need_scale) {
		std::cerr << "start scale data" << std::endl;
		std::vector<double> target_min(D), target_max(D);
		for (int d = 0; d < D; ++d) {
			target_min[d] = tmp_target_min;
			target_max[d] = tmp_target_max;
		}
		KMeans_scale_data(data, target_min, target_max, D);
	}
	KMeans_write_data(output_file_name, data, label, D, have_label, output_libsvm_format);

	return 0;
}
