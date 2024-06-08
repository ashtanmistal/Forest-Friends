#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

struct Point {
    float x, y, z;
};

void removeDEMHeight(std::vector<Point>& points, const cv::Mat& dem);
