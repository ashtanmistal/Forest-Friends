cmake_minimum_required(VERSION 3.10)
project(ForestFriends)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)

find_package(CUDA REQUIRED)
find_package(PDAL REQUIRED)

include_directories(${CMAKE_SOURCE_DIR} ${PDAL_INCLUDE_DIRS})

# Link directories
link_directories(${CMAKE_CUDA_TOOLKIT_ROOT_DIR}/lib64)

# Source files
set(SOURCES main.cpp clustering.cpp)

# Add executable
cuda_add_executable(ForestFriends ${SOURCES})

# Link libraries
target_link_libraries(ForestFriends cuml++)
