#ifndef PREPROCESSING_HPP
#define PREPROCESSING_HPP

#include <string>
#include <vector>
#include <functional>
#include "common.hpp"

int KMEANS_read_data(const std::string &file_name, DataMat &data, LabelVec &label, int N, int D, bool have_label = false, bool libsvm_format = false);
int KMEANS_write_data(std::vector<std::reference_wrapper<const std::string>> file_names, const DataMat &data, const LabelVec &label, bool have_label = false, bool libsvm_format = false);
int KMEANS_scale_data(DataMat &data, Eigen::VectorXf &target_min, Eigen::VectorXf &targe_max);

#endif
