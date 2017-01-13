#include <boost/program_options.hpp>
#include <boost/program_options/parsers.hpp>
#include <iostream>
#include <string>
#include <memory>
#include <Eigen/Dense>
#include <fstream>
#include <iomanip>

namespace po = boost::program_options;

struct config {
	std::string control_file_name;
	std::string output_file_name;
	std::string input_file_name;
	int precision;
	int data_number;
	int input_dimension;
	int output_dimension;
};

template <class T>
void read_file_line(std::shared_ptr<config> conf, std::ifstream &fin, Eigen::ArrayXi &need_output, T &vec) {
	for (int i = 0; i < conf->input_dimension; ++i) {
		typename T::Scalar v;
		fin >> v;
		if (need_output(i))
			vec(i) = v;
	}
}

void read_control_file(std::shared_ptr<config> conf, Eigen::ArrayXi &need_output, Eigen::ArrayXf &input_min, Eigen::ArrayXf &input_max, Eigen::ArrayXf &output_min, Eigen::ArrayXf &output_max) {
	std::ifstream fin(conf->control_file_name);
	int output_dimension = 0;
	for (int k = 0; k < conf->input_dimension; ++k) {
		fin >> need_output(k);
		if (need_output(k))
			++output_dimension;
	}
	read_file_line(conf, fin, need_output, input_min);
	read_file_line(conf, fin, need_output, input_max);
	read_file_line(conf, fin, need_output, output_min);
	read_file_line(conf, fin, need_output, output_max);
	fin.close();
}

void read_csv(std::shared_ptr<config> conf, Eigen::ArrayXXf &data, Eigen::ArrayXi &need_output) {
	std::ifstream fin(conf->input_file_name);
	for (int n = 0; n < conf->data_number; ++n) {
		float v;
		for (int i = 0; i < conf->input_dimension; ++i) {
			fin >> v;
			if (need_output(i))
				data(i, n) = v;
		}
	}
	fin.close();
}

void output_data(std::shared_ptr<config> conf, Eigen::ArrayXXf &data) {
	std::ofstream fout(conf->output_file_name);
	fout << std::fixed << std::setprecision(conf->precision);
	for (int n = 0; n < conf->data_number; ++n) {
		for (int k = 0; k < conf->output_dimension; ++k)
			fout << data(k, n) << ' ';
		fout << '\n';
	}
	fout.close();
}

int main(int argc, const char *argv[]) {
	std::shared_ptr<config> conf(new config);
	po::options_description desc("Allowed options");
	desc.add_options()
		("help,h", "produce help message")
		("control_file_name,c", po::value<std::string>(&conf->control_file_name)->default_value("control.txt"), "control file for scaling input data")
		("input_file_name,i", po::value<std::string>(&conf->input_file_name)->default_value("data.in"), "input file name")
		("output_file_name,o", po::value<std::string>(&conf->output_file_name)->default_value("data.out"), "output file name")
		("precision,p", po::value<int>(&conf->precision)->default_value(6), "number of digits in output file")
		("data_number,n", po::value<int>(&conf->data_number), "number of data point")
		("input_dimension,x", po::value<int>(&conf->input_dimension), "dimension of input data");
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);
	if (vm.count("help")) {
		std::cout << desc << std::endl;
		return 0;
	}
	if (!vm.count("data_number")) {
		std::cout << "must specify number of data points" << std::endl;
		std::cout << desc << std::endl;
		return 0;
	}
	if (!vm.count("data_dimension")) {
		std::cout << "must specify dimension of data points" << std::endl;
		std::cout << desc << std::endl;
		return 0;
	}

	Eigen::ArrayXi need_output(conf->data_number);
	Eigen::ArrayXf input_min(conf->data_number), input_max(conf->data_number), output_min(conf->data_number), output_max(conf->data_number);
	read_control_file(conf, need_output, input_min, input_max, output_min, output_max);
	Eigen::ArrayXXf data(conf->output_dimension, conf->data_number);
	read_csv(conf, data, need_output);

	for (int n = 0; n < conf->data_number; ++n)
		data.col(n) = output_min + (data.col(n) - input_min) * (input_max - input_min) * (output_max - output_min);
	output_data(conf, data);

	return 0;

}

