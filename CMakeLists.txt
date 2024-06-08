# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required(VERSION 3.18)

list(APPEND CMAKE_PREFIX_PATH "C:/dev/vcpkg/packages/curl_x64-windows/share/curl/")
set(OpenCV_DIR "C:/opencv/build")
# Note: VCPKG defaults in visual studio are not working so this is a workaround.
set(GDAL_DIR "C:/dev/vcpkg/packages/gdal_x64-windows/share/gdal")
set(PDAL_DIR "C:/dev/vcpkg/packages/pdal_x64-windows/share/pdal")

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project("Forest-Friends" LANGUAGES CUDA CXX)

find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-g;-G")
include_directories(${CUDA_INCLUDE_DIRS} ForestFriends/cuda)

# Include sub-projects.
add_subdirectory("ForestFriends")
