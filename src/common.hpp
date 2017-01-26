#ifndef KMeans_COMMON_HPP
#define KMeans_COMMON_HPP

#include <string>
#include <memory>
#include <Eigen/Dense>

struct KMeans_config {
	std::string data_file_name;
	std::string output_file_name;

	std::string input_seed_file_name;
	std::string output_seed_file_name;
	bool input_seed;
	bool output_seed;

	int data_number;
	int data_dimension;
	int cluster_number;

	int max_interation;
	bool until_converge;
	float norm_precision;

	bool kmeans_plus_plus_initialization;
	bool yinyang;
	int group_number;

	bool debug;
};

using DataMat = Eigen::MatrixXf;
using LabelVec = Eigen::VectorXf;
using CenterMat = Eigen::MatrixXf;
using ClusterVec = Eigen::VectorXi;
using PConf = std::shared_ptr<KMeans_config>;

std::ostream& operator<<(std::ostream& os, const KMeans_config& kc);
PConf KMeans_parse_arg(int argc, const char *argv[]);
int KMeans_get_data(PConf conf, DataMat &data, LabelVec &label);
//int KMeans_get_seed(PConf conf, ClusterVec &cluster);
int generate_libsvm_data_file(std::string file_name, PConf conf, DataMat &data, LabelVec &label);
int generate_random_initial_cluster(PConf conf, DataMat &data, CenterMat &center);
int output_cluster(PConf conf, ClusterVec &cluster);
double compute_loss(const DataMat &data, const ClusterVec &cluster, const CenterMat &center);
int KMeans_export_seed(std::string &file_name, CenterMat &center);
int KMeans_load_seed(std::string &file_name, int &K, int &D, CenterMat &center);

#endif
