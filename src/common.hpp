#ifndef KNN_COMMON_H
#define KNN_COMMON_H

#include <string>
#include <memory>
#include <Eigen/Dense>

struct KNN_config {
	std::string data_file_name;
	std::string output_file_name;
	int data_number;
	int data_dimension;
	int cluster_number;
	bool have_seed_file;
	int max_interation;
	float norm_precision;
	std::string seed_file_name;
};

std::ostream& operator<<(std::ostream& os, const KNN_config& kc);
std::shared_ptr<KNN_config> KNN_parse_arg(int argc, const char *argv[]);
int KNN_get_data(std::shared_ptr<KNN_config> conf, Eigen::MatrixXf &data, Eigen::VectorXf &label);
int KNN_get_seed(std::shared_ptr<KNN_config> conf, Eigen::VectorXi &cluster);
void generate_libsvm_data_file(std::string file_name, std::shared_ptr<KNN_config> conf, Eigen::MatrixXf &data, Eigen::VectorXf &label);
int generate_random_initial_cluster(std::shared_ptr<KNN_config> conf, Eigen::VectorXi &cluster);
void output_cluster(std::shared_ptr<KNN_config>conf, Eigen::VectorXi cluster);

#endif
