#ifndef CLUSTERING_H
#define CLUSTERING_H

#include <vector>

struct Point {
    float x, y, z;
};

std::vector<int> hdbscan(const std::vector<Point>& points);

#endif
