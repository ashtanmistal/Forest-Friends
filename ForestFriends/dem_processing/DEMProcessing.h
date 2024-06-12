#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

#include "Common.h"

void removeDEMHeight(std::vector<Point>& points, const cv::Mat& dem);
