﻿# CMakeList.txt : CMake project for ForestFriends, include source and define
# project specific logic here.
#

# Add source to this project's executable.
add_executable (ForestFriends "ForestFriends.cpp" "ForestFriends.h")
set(OpenCV_DIR "C:/opencv/build")
FIND_PACKAGE( OpenCV REQUIRED )                              
INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} )
find_package(GDAL CONFIG REQUIRED)

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET ForestFriends PROPERTY CXX_STANDARD 20)
endif()

# TODO: Add tests and install targets if needed.

target_link_libraries(ForestFriends PRIVATE GDAL::GDAL)