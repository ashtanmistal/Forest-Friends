#include "clustering.h"
#include <cuml/cluster/hdbscan.hpp>
#include <rmm/device_uvector.hpp>
#include <thrust/device_vector.h>

std::vector<int> performHDBSCAN(const std::vector<Point>& points) {
// TODO: Test this implementation( in backlog due to cuML library)
    size_t numPoints = points.size();
    thrust::device_vector<float> X(numPoints * 2);
    for (size_t i = 0; i < numPoints; i++) {
        X[i * 2] = points[i].x;
        X[i * 2 + 1] = points[i].y;
    }

    rmm::device_uvector<int> labels(numPoints, rmm::cuda_stream_default);
    rmm::device_uvector<int> nClusters(1, rmm::cuda_stream_default);
    rmm::device_uvector<int> sizes(numPoints, rmm::cuda_stream_default);
    rmm::device_uvector<int> treeSizes(numPoints, rmm::cuda_stream_default);

    cuml::HDBSCANParams params; // TODO: hyperparameter tuning and visualization of clusters
    params.minPts = 5;
    params.minSamples = 5;
    params.cluster_selection_epsilon = 0.0;
    params.allow_single_cluster = false;
    params.allow_multi_cluster = true;
    params.verbose = false;

    cuml::cluster::HDBSCAN hdbscan(params);
    hdbscan.fit(X.data().get(), numPoints, 2, labels.data().get(), nClusters.data().get(), sizes.data().get(), treeSizes.data().get());

    std::vector<int> result(numPoints);
    cudaMemcpy(result.data(), labels.data().get(), numPoints * sizeof(int), cudaMemcpyDeviceToHost);
    return result;
}
