#include "preprocessing.hpp"
#include <iostream>
#include <fstream>

int KMEANS_read_data(const std::string &file_name, DataMat &data, LabelVec &label, int N, int D, bool have_label = false, bool libsvm_format = false) {
	data.resize(D, N);
	if (have_label)
		label.resize(N);
	std::ifstream fin(file_name.c_str());
	for (int n = 0; n < N; ++n) {
		char c;
		if (have_label) {
			fin >> label(n);
			fin.get(c);
		}
		if (libsvm_format)
			while (true) {
				fin.get(c);
				if ('\n' == c)
					break;
				fin.unget();
				int d;
				float v;
				fin >> d >> c >> v;
				data(d - 1, n) = v;
			}
		else {
			for (int d = 0; d < D; ++d)
				fin >> data(d, n);
		}
	}
	fin.close();

	return 0;
}

int KMEANS_write_data_csv(std::vector<const std::string *> file_names, const DataMat &data, const LabelVec &label, bool have_label = false, bool libsvm_format = false) {
	int N = data.cols();
	int D = data.rows();

	std::ofstream fout;
	for (auto file_name: file_names) {
		fout.open(file_name.c_str());
		for (int n = 0; n < N; ++n) {
			if (have_label)
				fout << label(n) << ' ';
			if (libsvm_format)
				for (int d = 0; d < D; ++d)
					fout << d << ':' << data(d, n);
			else
				for (int d = 0; d < D; ++d)
					fout << data(d, n) << ' ';
			fout << '\n';
		}
		fout.close();
	}
}

int KMEANS_scale_data(DataMat &data, Eigen::VectorXf &target_min, Eigen::VectorXf &targe_max) {
	int D = data.rows();
	Eigen::VectorXf input_min(D), input_max(D);
	input_min = data.rowwise().minCoeff();
	input_max = data.rowwise().maxCoeff();

}
