#include "cuda_functions.h"
#include <cuda_runtime.h>
#include <iostream>


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
    removeDEMHeightKernel << <numBlocks, blockSize >> > (d_points, numPoints, d_demData, demWidth, demHeight, demXScale, demYScale, demXOffset, demYOffset);
    cudaDeviceSynchronize();

    cudaMemcpy(points.data(), d_points, numPoints * sizeof(Point), cudaMemcpyDeviceToHost);

    cudaFree(d_points);
    cudaFree(d_demData);
}