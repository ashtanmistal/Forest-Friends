#include <iostream>
#include <vector>
#include <filesystem>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <pdal/PointTable.hpp>
#include <pdal/PointView.hpp>
#include <pdal/StageFactory.hpp>
#include <pdal/io/LasReader.hpp>
#include <gdal_priv.h>
#include "clustering.h"
#include <cuda_runtime.h>

const std::string lidarDataDir = "data/";
const std::string demFile = lidarDataDir + "dem.tiff";

struct Point {
    float x, y, z;
};

// Load DEM using GDAL
cv::Mat loadDEM(const std::string& demFile) {
    GDALAllRegister();
    GDALDataset* dataset = (GDALDataset*)GDALOpen(demFile.c_str(), GA_ReadOnly);
    if (dataset == nullptr) {
        throw std::runtime_error("Failed to open DEM file.");
    }

    GDALRasterBand* band = dataset->GetRasterBand(1);
    int width = band->GetXSize();
    int height = band->GetYSize();
    cv::Mat dem(height, width, CV_32F);
    band->RasterIO(GF_Read, 0, 0, width, height, dem.data, width, height, GDT_Float32, 0, 0);

    GDALClose(dataset);
    return dem;
}

// CUDA kernel to remove DEM height
__global__ void removeDEMHeightKernel(Point* points, int numPoints, float* demData, int demWidth, int demHeight, float demXScale, float demYScale, float demXOffset, float demYOffset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPoints) {
        Point& point = points[idx];
        int demX = (point.x - demXOffset) / demXScale;
        int demY = (point.y - demYOffset) / demYScale;
        if (demX >= 0 && demX < demWidth && demY >= 0 && demY < demHeight) {
            int demIdx = demY * demWidth + demX;
            point.z -= demData[demIdx];
        }
    }
}


void removeDEMHeight(std::vector<Point>& points, const cv::Mat& dem) {
    int numPoints = points.size();
    int demWidth = dem.cols;
    int demHeight = dem.rows;
    // TODO set the offsets correctly
    float demXScale = 1.0f;
    float demYScale = 1.0f;
    float demXOffset = 0.0f; // this needs to be set to the min x and y of the overall dataset, IIRC
    float demYOffset = 0.0f;

    Point* d_points;
    float* d_demData;
    cudaMalloc(&d_points, numPoints * sizeof(Point));
    cudaMalloc(&d_demData, demWidth * demHeight * sizeof(float));

    cudaMemcpy(d_points, points.data(), numPoints * sizeof(Point), cudaMemcpyHostToDevice);
    cudaMemcpy(d_demData, dem.ptr<float>(), demWidth * demHeight * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256; // TODO find optimal block size based on VRAM available. 
    int numBlocks = (numPoints + blockSize - 1) / blockSize;
    removeDEMHeightKernel<<<numBlocks, blockSize>>>(d_points, numPoints, d_demData, demWidth, demHeight, demXScale, demYScale, demXOffset, demYOffset);
    cudaDeviceSynchronize();

    cudaMemcpy(points.data(), d_points, numPoints * sizeof(Point), cudaMemcpyDeviceToHost);

    cudaFree(d_points);
    cudaFree(d_demData);
}

void processDataset(const std::string& inputFile, const std::string& outputFile) {
    std::cout << "Processing file: " << inputFile << std::endl;

    pdal::StageFactory factory;
    pdal::Options options;
    options.add("filename", inputFile);

    pdal::LasReader reader;
    reader.setOptions(options);

    pdal::PointTable table;
    reader.prepare(table);
    pdal::PointViewSet viewSet = reader.execute(table);

    pdal::PointViewPtr view = *viewSet.begin();

    size_t numPoints = view->size();
    std::vector<Point> points(numPoints);
    for (pdal::PointId i = 0; i < numPoints; ++i) {
        points[i] = {
            view->getFieldAs<float>(pdal::Dimension::Id::X, i),
            view->getFieldAs<float>(pdal::Dimension::Id::Y, i),
            view->getFieldAs<float>(pdal::Dimension::Id::Z, i)
        };
    }
    cv::Mat dem = loadDEM(demFile);
    removeDEMHeight(points, dem);

    // TODO Perform clustering using HDBSCAN
    std::vector<int> labels = performHDBSCAN(points); // Dummy function; implement in clustering.cpp

    // Visualization
    cv::Mat image(1000, 1000, CV_8UC3, cv::Scalar(255, 255, 255));
    for (size_t i = 0; i < points.size(); ++i) {
        int x = static_cast<int>(points[i].x * 10) % 1000;
        int y = static_cast<int>(points[i].y * 10) % 1000;
        cv::circle(image, cv::Point(x, y), 1, cv::Scalar(0, 0, 255), -1);
    }

    std::string outputImageFile = outputFile.substr(0, outputFile.find_last_of('.')) + ".png";
    cv::imwrite(outputImageFile, image);
    // std::cout << "Saved clustered image to " << outputImageFile << std::endl;

    // Save cluster information
    std::ofstream clusterFile(outputFile);
    for (size_t i = 0; i < points.size(); ++i) {
        clusterFile << points[i].x << " " << points[i].y << " " << points[i].z << " " << labels[i] << "\n";
    }
    clusterFile.close();
    // std::cout << "Saved cluster information to " << outputFile << std::endl;
}

int main() {
    for (const auto& entry : std::filesystem::directory_iterator(lidarDataDir)) {
        if (entry.path().extension() == ".las") {
            std::string inputFile = entry.path().string();
            std::string outputFile = entry.path().stem().string() + ".clustered";
            processDataset(inputFile, outputFile);
        }
    }
    return 0;
}