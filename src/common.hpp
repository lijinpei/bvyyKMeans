#ifndef KMeans_COMMON_HPP
#define KMeans_COMMON_HPP

#include <string>
#include <memory>
#include <vector>

#include <iostream>
#include <fstream>
#include <iomanip>

#include <ctime>
#include <cstdint>
#include <cstdio>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_sparse.hpp>
#include <boost/program_options.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/random.hpp>

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
	int block_size;

	bool debug;
	bool sparse;
};

template <class T>
using DataMat = std::vector<T>;
using LabelVec = std::vector<float>;
template <class T>
using CenterMat = std::vector<T>;
using ClusterVec = std::vector<int>;
using PConf = std::shared_ptr<KMeans_config>;
template <class T>
using SparseVec = boost::numeric::ublas::compressed_vector<T>;
template <class T>
using DenseVec = boost::numeric::ublas::vector<T>;

std::ostream& operator<<(std::ostream& os, const KMeans_config& kc);
PConf KMeans_parse_arg(int argc, const char *argv[]);

template <class T>
inline double bvyyKMeansNorm(const T &v) {
/*
	double n = 0;
	for (auto a: v)
		n += a * a;
	return std::sqrt(n);
*/
	return boost::numeric::ublas::norm_2(v);
}

template <class T>
inline double bvyyKMeansSquaredNorm(const T &v) {
/*
	double n = 0;
	for (auto a: v)
		n += a * a;
	return std::sqrt(n);
*/
	double tmp_v = boost::numeric::ublas::norm_2(v);
	return tmp_v * tmp_v;
}

template <class T>
inline double bvyyKMeansDistance(const T&v1, const T&v2) {
	return bvyyKMeansNorm(v1 - v2);
}

template <class T>
inline double bvyyKMeansSquaredDistance(const T&v1, const T&v2) {
	return bvyyKMeansSquaredNorm(v1 - v2);
}

template <class T, class V = float>
inline void bvyyKMeansInsert(T& vec, int d, V val) {
	vec(d) = val;
}

template <class T>
inline double bvyyKMeansInnerProduct(const T&v1, const T&v2) {
	return boost::numeric::ublas::inner_prod(v1, v2);
}

template <class T>
int KMeans_get_data(PConf conf, DataMat<T> &data, LabelVec &label) {
	FILE* f = fopen(conf->data_file_name.c_str(), "r");
	const int& N = conf->data_number;
	const int& D = conf->data_dimension;
	bool err = false;
	for (int n = 0; n < N; ++n) {
		if (1 != fscanf(f, "%f%*[ ]", &label[n])) {
			err = true;
			break;
		}
		int c;
		while ('\n' != (c = getc(f))) {
			ungetc(c, f);
			int i;
			float v;
			if (2 != fscanf(f, "%d:%f%*[ ]", &i, &v) || i > D) {
				err = true;
				break;
			}
			bvyyKMeansInsert(data[n], i - 1 , v);
		}
	}
	fclose(f);
	if (err) {
		std::cerr << "Error in reading data file" << std::endl;
		return -1;
	}
	return 0;
}

template <class T>
int generate_libsvm_data_file(std::string file_name, PConf conf, DataMat<T> &data, LabelVec &label) {
	std::ofstream fout(file_name.c_str());
	fout << std::fixed << std::setprecision(6);
	const int &N = conf->data_number;
	const int &D = conf->data_dimension;
	for (int n = 0; n < N; ++n) {
		fout << int(label[n]) << ' ';
		for (int d = 0; d < D; ++d)
			fout << d+1 << ":" << data[n][d] << ' ';
		fout << '\n';
	}
	fout.close();

	return 0;
}

template <class T>
int generate_random_initial_cluster(PConf conf, DataMat<T> &data, CenterMat<T> &center) {
	boost::random::mt19937 gen{static_cast<std::uint32_t>(std::time(0))};
	boost::random::uniform_int_distribution<> dist{0, conf->data_number - 1};
	std::vector<int> chosen(conf->data_number);
	for (int i = 0; i < conf->cluster_number; ++i) {
		int n;
		while (true) {
			n = dist(gen);
			if (!chosen[n]) {
				chosen[n] = 1;
				break;
			}
		}
		center[i] = data[n];
	}
	return 0;
}

int output_cluster(PConf conf, ClusterVec &cluster);

template <class T>
double compute_loss(const DataMat<T> &data, const ClusterVec &cluster, const CenterMat<T> &center) {
	//std::cerr << "start compute loss" << std::endl;
	double l = 0;
	const int N = data.size();
	for (int n = 0; n < N; ++n) {
		/*
		std::cerr << "n " << n << std::endl;
		std::cerr << "cluster " << std::endl << cluster << std::endl;
		std::cerr << "cluster(n) " << cluster(n) << std::endl;
		*/
		l += bvyyKMeansDistance(data[n], center[cluster[n]]);
	}
	//std::cerr << "finished compute loss" << std::endl;
	return l;
}

template <class T>
int KMeans_export_seed(std::string &file_name, CenterMat<T> &center, const int K, const int D) {
	std::ofstream fout(file_name.c_str(), std::ios::out|std::ios::binary);
	// warning: don't do this across machines
	fout.write(reinterpret_cast<const char*>(&K), sizeof(int));
	fout.write(reinterpret_cast<const char*>(&D), sizeof(int));
	for (int k = 0; k < K; ++k)
		for (int d = 0; d < D; ++d) {
			double tmp_v = center[k][d];
			fout.write(reinterpret_cast<const char*>(&tmp_v), sizeof(double));
		}
	fout.close();

	return 0;
}

template <class T>
int KMeans_load_seed(std::string &file_name, int &K, int &D, CenterMat<T> &center) {
	std::ifstream fin(file_name.c_str(), std::ios::in|std::ios::binary);
	// warning: don't do this across machines
	fin.read(reinterpret_cast<char*>(&K), sizeof(int));
	fin.read(reinterpret_cast<char*>(&D), sizeof(int));
	std::cerr << "K: " << K << " D: " << D << std::endl;
	center.resize(K);
	for (int k = 0; k < K; ++k) {
		for (int d = 0; d < D; ++d) {
			double tmp_v;
			fin.read(reinterpret_cast<char*>(&tmp_v), sizeof(double));
			bvyyKMeansInsert(center[k], d, tmp_v);
		}
	}
	fin.close();

	return 0;
}

#endif

