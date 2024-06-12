// ForestFriends.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include "Common.h"

#include <iostream>
#include <vector>
#include <filesystem>
#include <fstream>
#include <string>
#include <stdexcept>

#include <gdal.h>
#include <gdal_priv.h>
#include <cpl_conv.h>

#include <pdal/pdal.hpp>
#include <pdal/StageFactory.hpp>
#include <pdal/Options.hpp>
#include <pdal/PointTable.hpp>
#include <pdal/PointView.hpp>
#include <pdal/io/LasReader.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


cv::Mat loadDEM(const std::string& demFile);
void processDataset(const std::string& inputFile, const std::string& outputFile);
