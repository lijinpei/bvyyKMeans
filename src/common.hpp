#ifndef KNN_COMMON_H
#define KNN_COMMON_H

#include <string>
#include <memory>

struct KNN_config {
	std::string data_file_name;
	int data_number;
	int data_dimension;
	int cluster_number;
	bool have_seed_file;
	std::string seed_file_name;
};

std::ostream& operator<<(std::ostream& os, const KNN_config& kc);
std::unique_ptr<KNN_config> KNN_parse_arg(int argc, const char *argv[]);

#endif
