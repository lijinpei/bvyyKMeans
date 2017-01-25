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
	std::string control_file_name;

	int D;
	po::options_description desc("Allowed options");
	desc.add_options()
		("help,h", "produce help message")
		("input_file,i", po::value<std::vector<std::string>>(&input_file_names)->multitoken(), "name of input files")
		("output_file,o", po::value<std::string>(&output_file_name), "output file")
		("input_libsvm,s", po::value<bool>(&input_libsvm_format)->default_value(false), "switch on this option to read input data in libsvm format")
		("output_libsvm,m", po::value<bool>(&output_libsvm_format)->default_value(false), "switch on this option to write output data in libsvm format")
		("scale,s", po::value<bool>(&need_scale)->default_value(false), "switch on this option to scale data")
		("control,c", po::value<std::string>(&control_file_name), "control file for transforming data")
		("label,l",po::value<bool>(&have_label)->default_value(false), "switch on this option to process data file with label")
		("d", po::value<int>(&D), "dimension of input data");
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);
	if (vm.count("help")) {
		std::cout << desc << "\n";
		return 0;
	}

	std::vector<int> control;
	int tmp_target_min = -1, tmp_target_max = 1;
	if (vm.count("control")) {
		std::ifstream control_file(control_file_name.c_str());
		for (int d = 0; d < D; ++d)
			control_file >> control[d];
		control_file >> tmp_target_min;
		control_file >> tmp_target_max;
	}



	std::vector<std::vector<double>> data;
	std::vector<double> label;
	for (auto file_name: input_file_names)
		KMEANS_read_data(file_name, data, label, D, have_label, input_libsvm_format);
	if (need_scale) {
		std::vector<double> target_min(D), target_max(D);
		for (int d = 0; d < D; ++d) {
			target_min[d] = tmp_target_min;
			target_max[d] = tmp_target_max;
		}
		KMEANS_scale_data(data, target_min, target_max, D);
	}
	KMEANS_write_data(output_file_name, data, label, D, have_label, output_libsvm_format);

	return 0;
}
