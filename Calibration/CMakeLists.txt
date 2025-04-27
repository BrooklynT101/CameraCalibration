CMAKE_MINIMUM_REQUIRED(VERSION "3.12")

set(CMAKE_CXX_STANDARD 14)
#this is set manually for the lab computers this should not be required for other systems with proper installation
set(CMAKE_TOOLCHAIN_FILE "C:\\vcpkg\\scripts\\buildsystems\\vcpkg.cmake")


project(Calibration)
find_package(OpenCV REQUIRED)#find_package (OpenCV CONFIG REQUIRED)
include_directories (${OpenCV_INCLUDE_DIRS})
add_executable(Calibration calibration.cpp)
target_link_libraries(Calibration
  ${OpenCV_LIBS}
)