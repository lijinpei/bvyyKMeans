#include "common.hpp"
#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
	auto conf = KMEANS_parse_arg(argc, argv);
	if (!conf)
		return 0;
	std::cout << *conf << std::endl;

	return 0;
}
