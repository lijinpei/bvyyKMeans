#ifndef LLOYD_H
#define LLOYD_H
#include "common.hpp"

bool naive_update_center(DataMat &data, ClusterVec &cluster, CenterMat &center, double precision, Eigen::MatrixXd &workspace1, Eigen::VectorXi &workspace2);
bool naive_update_cluster(DataMat &data, ClusterVec &cluster, CenterMat &center);
int lloyd(const DataMat &data, ClusterVec &cluster, CenterMat &center, double precision, int max_interation, bool until_converge);

#endif
