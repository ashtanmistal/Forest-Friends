#include "clustering.h"
#include <cuml/cluster/hdbscan.hpp>
#include <rmm/device_uvector.hpp>
#include <thrust/device_vector.h>

std::vector<int> performHDBSCAN(const std::vector<Point>& points) {
	size_t numPoints = points.size();

}