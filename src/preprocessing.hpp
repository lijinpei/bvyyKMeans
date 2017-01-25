#ifndef PREPROCESSING_HPP
#define PREPROCESSING_HPP

#include <string>
#include <vector>

int KMEANS_read_data(const std::string &file_name, std::vector<std::vector<double>> &data, std::vector<double> &label, int D, bool have_label = false, bool libsvm_format = false);
int KMEANS_write_data(const std::string file_names, std::vector<std::vector<double>> &data, std::vector<double> &label, int D, bool have_label = false, bool libsvm_format = false);
int KMEANS_scale_data(std::vector<std::vector<double>> &data, std::vector<double> &target_min, std::vector<double> &targe_max, int D);

#endif
