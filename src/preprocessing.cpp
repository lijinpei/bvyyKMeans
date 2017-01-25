#include "preprocessing.hpp"
#include <iostream>
#include <fstream>
#include <vector>

int KMEANS_read_data(const std::string &file_name, std::vector<std::vector<double>> &data, std::vector<double> &label, int D, bool have_label, bool libsvm_format) {
	std::ifstream fin(file_name.c_str());
	while (EOF != fin.peek()) {
		char c;
		if (have_label) {
			double tmp_d;
			fin >> tmp_d;
			label.push_back(tmp_d);
			fin.get(c);
		}
		std::vector<double> tmp_v(D);
		if (libsvm_format)
			while (true) {
				fin.get(c);
				if ('\n' == c)
					break;
				fin.unget();
				int d;
				double v;
				fin >> d >> c >> v;
				fin.get(c);
				tmp_v[d - 1] = v;

			}
		else {
			for (int d = 0; d < D; ++d) {
				double v;
				fin >> v >> c;
				tmp_v[d] = v;
			}
			fin.get(c);
		}
		data.push_back(tmp_v);
	}
	fin.close();

	return 0;
}

int KMEANS_write_data(std::string file_name, std::vector<std::vector<double>> &data, std::vector<double> &label, int D, bool have_label, bool libsvm_format) {
	std::cerr << "start output to file " << file_name << std::endl;
	std::ofstream fout(file_name.c_str());
	int N = data.size();
	std::cerr << "data number " << N << std::endl;
	std::cerr << "data dimension " << D << std::endl;
	for (int n = 0; n < N; ++n) {
		if (have_label)
			fout << label[n] << ' ';
		if (libsvm_format) {
			for (int d = 0; d < D; ++d) {
				fout << d + 1 << ':' << data[n][d] << ' ';
			}
		} else {
			for (int d = 0; d < D; ++d) {
				fout << data[n][d] << ' ';
			}
		}
		fout << '\n';
	}
	fout.close();

	return 0;
}

int KMEANS_scale_data(std::vector<std::vector<double>> &data, std::vector<double> &target_min, std::vector<double> &target_max, int D) {
	std::vector<double> input_min(D), input_max(D);
	for (int d = 0; d < D; ++d) {
		input_min[d] = data[0][d];
		input_max[d] = data[0][d];
	}
	for (auto d: data) {
		for (int i = 0; i < D; ++i) {
			double tmp_d = d[i];
			if (tmp_d < input_min[i])
				input_min[i] = tmp_d;
			else if (tmp_d > input_max[i]) 
				input_max[i] = tmp_d;
		}
	}
	int N = data.size();
	for (int n = 0; n < N; ++n) {
		for (int d = 0; d < D; ++d) {
			double imin = input_min[d], imax = input_max[d];
			double tmin = target_min[d], tmax = target_max[d];
			data[n][d] = (data[n][d] - imin) / (imax - imin) * (tmax - tmin) + tmin;
		}
	}

	return 0;
}
