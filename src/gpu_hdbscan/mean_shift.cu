#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void meanShiftKernel(float* data, int dataSize, float bandwidth, float* result) {
    // Get the index of the current thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if the thread index is within the data size
    if (idx < dataSize) {
        // Initialize convergence threshold
        float convergenceThreshold = 0.001f;  // TODO we can set this as a param
        
        // Initialize mean shift vector
        float meanShiftX = 0.0f;
        float meanShiftY = 0.0f;

        int maxIterations = 100;   // TODO we should set this as a param
        int iterationCount = 0;
        
        // Iterate until convergence
        bool converged = false;
        while (!converged) {
            // Compute mean shift vector for the current data point
            float totalWeight = 0.0f;
            float sumShiftX = 0.0f;
            float sumShiftY = 0.0f;
            for (int i = 0; i < dataSize; ++i) {
                float diffX = data[i * 2] - data[idx * 2];
                float diffY = data[i * 2 + 1] - data[idx * 2 + 1];
                float distanceSquared = diffX * diffX + diffY * diffY;
                float weight = exp(-distanceSquared / (2 * bandwidth * bandwidth));
                sumShiftX += weight * diffX;
                sumShiftY += weight * diffY;
                totalWeight += weight;
            }
            meanShiftX = sumShiftX / totalWeight;
            meanShiftY = sumShiftY / totalWeight;
            
            // Check for convergence by computing the magnitude of the mean shift vector
            float meanShiftMagnitude = sqrt(meanShiftX * meanShiftX + meanShiftY * meanShiftY);
            converged = (meanShiftMagnitude < convergenceThreshold);
            
            // Update data point based on mean shift vector
            data[idx * 2] += meanShiftX;
            data[idx * 2 + 1] += meanShiftY;

            // Increment iteration count and break if we've reached the maximum number of iterations
            ++iterationCount;
            converged = (converged || (iterationCount >= maxIterations));
        }
        
        result[idx * 2] = data[idx * 2];
        result[idx * 2 + 1] = data[idx * 2 + 1];
    }
}

// estimate the bandwidth of the kernel
// using the median distance between points
__device__ float estimateBandwidth(float *data, int dataSize) {
    /*
    Calculate the bandwidth of a kernel using Silverman's rule of thumb.
    There are more accurate methods available and this exists as more of a backup.
    Should the user want a more accurate bandwidth it can be specified by the user
    */
    float varianceX = 0.0f;
    float varianceY = 0.0f;
    for (int i = 0; i < dataSize; ++i) {
        varianceX += (data[i * 2] * data[i * 2]) / dataSize;
        varianceY += (data[i * 2 + 1] * data[i * 2 + 1]) / dataSize;
    }
    float bandwidthX = 1.06f * sqrt(varianceX);
    float bandwidthY = 1.06f * sqrt(varianceY);
    float bandwidth = (bandwidthX + bandwidthY) / 2.0f;
    return bandwidth;
}


extern "C" void meanShiftCUDA(float* data, int dataSize, float bandwidth = -1.0f, int maxIterations, float convergenceThreshold, float* result) {
    float *dev_data, *dev_result;

    if (bandwidth < 0.0f) {
        bandwidth = estimateBandwidth(data, dataSize);
    }
    
    // Allocate memory on the GPU and copy over the data from the CPU
    cudaMalloc((void**)&dev_data, dataSize * sizeof(float) * 2);
    cudaMalloc((void**)&dev_result, dataSize * sizeof(float) * 2);
    cudaMemcpy(dev_data, data, dataSize * sizeof(float) * 2, cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int numBlocks = (dataSize + blockSize - 1) / blockSize;
    
    // send the kernel to the GPU and wait for it to finish
    meanShiftKernel<<<numBlocks, blockSize>>>(dev_data, dataSize, bandwidth, dev_result);
    cudaDeviceSynchronize();
    cudaMemcpy(result, dev_result, dataSize * sizeof(float) * 2, cudaMemcpyDeviceToHost);
    
    cudaFree(dev_data);
    cudaFree(dev_result);
}



