# CMakeList.txt : CMake project for ForestFriends, include source and define
# project specific logic here.
#

# Add source to this project's executable.
add_executable(ForestFriends "ForestFriends.cpp" "clustering/clustering.cpp" "dem_processing/DEMProcessing.cu")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(GDAL CONFIG REQUIRED)
include_directories(${GDAL_INCLUDE_DIR})

find_package(PDAL CONFIG REQUIRED)
target_include_directories(ForestFriends PRIVATE ${PDAL_INCLUDE_DIRS})

find_package(CUDAToolkit REQUIRED)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

target_link_libraries(ForestFriends PRIVATE ${OpenCV_LIBS} GDAL::GDAL ${PDAL_LIBRARIES} ${CUDAToolkit_LIBRARIES})

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET ForestFriends PROPERTY CXX_STANDARD 20)
endif()