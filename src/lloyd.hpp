#ifndef LLOYD_H
#define LLOYD_H
#include "common.hpp"

bool lloyd_update_center(const DataMat &data, const ClusterVec &cluster, CenterMat &center, double precision, Eigen::MatrixXd &workspace1, Eigen::VectorXi &workspace2);
bool lloyd_update_cluster(const DataMat &data, ClusterVec &cluster, const CenterMat &center);
int lloyd(const DataMat &data, ClusterVec &cluster, CenterMat &center, double precision, int max_interation, bool until_converge);

#endif
