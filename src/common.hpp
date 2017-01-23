#ifndef KMEANS_COMMON_H
#define KMEANS_COMMON_H

#include <string>
#include <memory>
#include <Eigen/Dense>

struct KMEANS_config {
	std::string data_file_name;
	std::string output_file_name;
	int data_number;
	int data_dimension;
	int cluster_number;
	bool have_seed_file;
	int max_interation;
	float norm_precision;
	bool until_converge;
	bool kmeans_plus_plus_initialization;
	std::string seed_file_name;
};

using DataMat = Eigen::MatrixXf;
using LabelVec = Eigen::VectorXf;
using CenterMat = Eigen::MatrixXf;
using ClusterVec = Eigen::VectorXi;
using PConf = std::shared_ptr<KMEANS_config>;

std::ostream& operator<<(std::ostream& os, const KMEANS_config& kc);
PConf KMEANS_parse_arg(int argc, const char *argv[]);
int KMEANS_get_data(PConf conf, DataMat &data, LabelVec &label);
int KMEANS_get_seed(PConf conf, ClusterVec &cluster);
void generate_libsvm_data_file(std::string file_name, PConf conf, DataMat &data, LabelVec &label);
int generate_random_initial_cluster(PConf conf, ClusterVec &cluster);
void output_cluster(PConf conf, ClusterVec &cluster);
double compute_loss(PConf conf, DataMat &data, ClusterVec &cluster, CenterMat &center);


int kmeans_plus_plus_initialize(PConf conf, DataMat &data, CenterMat &center);
#endif
