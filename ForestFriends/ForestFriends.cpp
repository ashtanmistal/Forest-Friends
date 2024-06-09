// ForestFriends.cpp : Defines the entry point for the application.
//

#include "ForestFriends.h"
#include "dem_processing/DEMProcessing.h"
#include "clustering/clustering.h"


const std::string lidarDataDir = "data/";
const std::string demFile = lidarDataDir + "dem.tiff";


// Load DEM using GDAL
cv::Mat loadDEM(const std::string& demFile) {
// Instead of using GDAL for this we could convert it into a more readable format in Python and then load it here
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
    std::vector<int> labels = performHDBSCAN(points);

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