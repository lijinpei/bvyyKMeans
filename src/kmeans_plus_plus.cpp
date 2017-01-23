#include "common.hpp"
#include <ctime>
#include <boost/random.hpp>
#include <Eigen/Dense>
#include <iostream>

int choose_center(PConf conf, Eigen::VectorXd &min_dist, Eigen::VectorXi &chosen) {
	double sum = min_dist.sum();
	static boost::random::mt19937 gen{static_cast<std::uint32_t>(std::time(0))};
	static boost::uniform_real<> uni_dist(0,1);
	static boost::variate_generator<boost::random::mt19937&, boost::uniform_real<> > uni(gen, uni_dist);
	double ran = uni() * sum;
	double psum = 0;
	for (int i = 0; i < conf->data_number; ++i)
		if (!chosen(i)) {
			psum += min_dist(i);
			if (psum >= ran)
				return i;
		}
	return -1;

}

int kmeans_plus_plus_initialize(PConf conf, DataMat &data, CenterMat &center) {
	static boost::random::mt19937 gen{static_cast<std::uint32_t>(std::time(0))};
	static boost::random::uniform_int_distribution<> dist{0, conf->data_number - 1};
	Eigen::VectorXi chosen(conf->data_number);
	chosen = Eigen::VectorXi::Zero(conf->data_number);
	int n =  dist(gen);
	center.col(0) = data.col(n);
	Eigen::VectorXd min_dist(conf->data_number);
	min_dist = (data.colwise() - center.col(0)).colwise().squaredNorm().cast<double>();
	for (int i = 1; i <conf->cluster_number; ++i) {
		n = choose_center(conf, min_dist, chosen);
		if (-1 == n)
			return -1;
		chosen(n) = 1;
		center.col(i) = data.col(n);

		std::cerr << min_dist.rows() << ' ' << min_dist.cols() << std::endl;
		std::cerr << (data.colwise() - center.col(i)).colwise().squaredNorm().rows() << ' ' << (data.colwise() - center.col(i)).colwise().squaredNorm().cols() << std::endl;
		min_dist.array().min((data.colwise() - center.col(i)).colwise().squaredNorm().transpose().cast<double>().array());
	}

	return 0;
}
