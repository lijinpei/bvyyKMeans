#ifndef KMeans_PLUS_PLUS_H
#define KMeans_PLUS_PLUS_H
#include "common.hpp"

template <class T>
int kmeans_plus_plus_initialize(const DataMat<T> &data, CenterMat<T> &center);
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

template <class T>
int kmeans_plus_plus_initialize(const DataMat<T> &data, CenterMat<T> &center) {
	int N = data.size();
	int K = center.size();
	std::cerr << "Number of data points in kmeans++ " << N << std::endl;
	std::cerr << "Number of groups in kmeans++ " << K << std::endl;
	boost::random::mt19937 gen{static_cast<std::uint32_t>(std::time(0))};
	boost::random::uniform_int_distribution<> dist(0, N - 1);
	ClusterVec chosen(N);
	int n =  dist(gen);
	center[0] = data[n];
	Eigen::VectorXd min_dist(N);
	min_dist = (data.colwise() - center.col(0)).colwise().squaredNorm().cast<double>();
	for (int i = 1; i < K; ++i) {
		n = choose_center(min_dist, chosen);
		if (-1 == n)
			return -1;
		chosen(n) = 1;
		center.col(i) = data.col(n);

		//min_dist.array().min((data.colwise() - center.col(i)).colwise().squaredNorm().transpose().cast<double>().array());
		for (int n = 0; n < N; ++n) {
			double nv = (data.col(n) - center.col(i)).squaredNorm();
			if (nv < min_dist(n))
				min_dist(n) = nv;
		}
		if (std::abs(min_dist(n)) > 1e-5) {
			std::cerr << "error in min_dist" << std::endl;
			return -1;
		}
	}

	return 0;
}

#endif
