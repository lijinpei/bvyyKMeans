#include "common.hpp"
#include <ctime>
#include <boost/random.hpp>
#include <Eigen/Dense>
#include <iostream>

int choose_center(Eigen::VectorXd &min_dist, Eigen::VectorXi &chosen) {
	double sum = min_dist.sum();
	static boost::random::mt19937 gen{static_cast<std::uint32_t>(std::time(0))};
	static boost::uniform_real<> uni_dist(0,1);
	static boost::variate_generator<boost::random::mt19937&, boost::uniform_real<> > uni(gen, uni_dist);
	double ran = uni() * sum;
	double psum = 0;
	int N = chosen.rows();
	for (int n = 0; n < N; ++n)
		if (!chosen(n)) {
			psum += min_dist(n);
			if (psum >= ran)
				return n;
		}
	return -1;

}

int kmeans_plus_plus_initialize(const DataMat &data, CenterMat &center) {
	int N = data.cols();
	int K = center.cols();
	static boost::random::mt19937 gen{static_cast<std::uint32_t>(std::time(0))};
	static boost::random::uniform_int_distribution<> dist{0, N - 1};
	Eigen::VectorXi chosen(N);
	chosen.setZero();
	int n =  dist(gen);
	center.col(0) = data.col(n);
	Eigen::VectorXd min_dist(N);
	min_dist = (data.colwise() - center.col(0)).colwise().squaredNorm().cast<double>();
	for (int i = 1; i < K; ++i) {
		n = choose_center(min_dist, chosen);
		if (-1 == n)
			return -1;
		chosen(n) = 1;
		center.col(i) = data.col(n);

		min_dist.array().min((data.colwise() - center.col(i)).colwise().squaredNorm().transpose().cast<double>().array());
	}

	return 0;
}
