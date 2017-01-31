#ifndef KMeans_PLUS_PLUS_HPP
#define KMeans_PLUS_PLUS_HPP
#include "common.hpp"
#include <vector>
#include <numeric>

int choose_center(const std::vector<double> &min_dist, std::vector<int> &chosen) {
	double sum = std::accumulate(min_dist.begin(), min_dist.end(), 0);
	static boost::random::mt19937 gen{static_cast<std::uint32_t>(std::time(0))};
	static boost::uniform_real<> uni_dist(0,1);
	static boost::variate_generator<boost::random::mt19937&, boost::uniform_real<> > uni(gen, uni_dist);
	double ran = uni() * sum;
	double psum = 0;
	const int N = chosen.size();
	for (int n = 0; n < N; ++n) {
		if (!chosen[n]) {
			psum += min_dist[n];
			if (psum >= ran)
				return n;
		}
	}
	return -1;

}

template <class T>
int kmeans_plus_plus_initialize(const DataMat<T> &data, CenterMat<T> &center) {
	const int N = data.size();
	const int K = center.size();
	//std::cerr << "Number of data points in kmeans++ " << N << std::endl;
	//std::cerr << "Number of groups in kmeans++ " << K << std::endl;
	boost::random::mt19937 gen{static_cast<std::uint32_t>(std::time(0))};
	boost::random::uniform_int_distribution<> dist(0, N - 1);
	ClusterVec chosen(N);
	int n =  dist(gen);
	center[0] = data[n];
	std::vector<double> min_dist(N);
	for (int n = 0; n < N; ++n)
		min_dist[n] = bvyyKMeansSquaredDistance(data[n], center[0]);
	for (int k = 1; k < K; ++k) {
		n = choose_center(min_dist, chosen);
		if (-1 == n)
			return -1;
		chosen[n] = 1;
		center[k] = data[n];

		//min_dist.array().min((data.colwise() - center.col(i)).colwise().squaredNorm().transpose().cast<double>().array());
		for (int n = 0; n < N; ++n) {
			double nv = bvyyKMeansSquaredDistance(data[n], center[k]);
			if (nv < min_dist[n])
				min_dist[n] = nv;
		}
		if (std::abs(min_dist[n] > 1e-5)) {
			std::cerr << "error in min_dist" << std::endl;
			return -1;
		}
	}

	return 0;
}

#endif
