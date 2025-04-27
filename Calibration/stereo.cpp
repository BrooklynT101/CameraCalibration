#include <opencv2/opencv.hpp>
#include "calibrationIO.h"

/* Create a method that will filter through the bell image pair,
 * through different levels of block sizing and number of disparities.
 * This will use the same images for each test but each one will change the 
 * number of disparities and the block size between the ranges of 16-128 and 5-21 respectively.
 * Each step will be 16 for the number of disparities and 2 for the block size, after the block matching
 * the resulting images will be saved to the disk, with descriptive names.
*/
int matcherValueTest() {
}


int main(int argc, char* argv[]) {

	const bool debugging = false;

	// Command line parameters
	// stereo <image1> <image2> <calibration>

	// Read in the two images
	cv::Mat image1 = cv::imread(argv[1]);
	cv::Mat image2 = cv::imread(argv[2]);

	// Read in the calibratin data
	cv::Mat K1, K2, d1, d2, R, t;
	readStereoCalibration(argv[3], K1, d1, K2, d2, R, t);

	// Output the calibration data as a check that they were read OK
	std::cout << "K1" << std::endl << K1 << std::endl;
	std::cout << "d1" << std::endl << d1 << std::endl;
	std::cout << "K2" << std::endl << K2 << std::endl;
	std::cout << "d2" << std::endl << d2 << std::endl;
	std::cout << "R" << std::endl << R << std::endl;
	std::cout << "t" << std::endl << t << std::endl;

	// Rectify the images
	/*
	* R1 – a transformation (rotation matrix) that moves 3D points from the original to rectified camera spaces for camera 1.
	* R2 – a transformation (rotation matrix) that moves 3D points from the original to rectified camera spaces for camera 2.
	* P1 – the projection matrix for the (virtual) camera view that would produce the first rectified image.
	* P2 – the projection matrix for the (virtual) camera view that would produce the second rectified image.
	* Q – the 3D transformation that converts an image point and associated disparity into a 3D point.
	*/
	cv::Mat R1, R2, P1, P2, Q;
	cv::stereoRectify(
		K1, d1, K2, d2,
		image1.size(), R, t,
		R1, R2, P1, P2, Q);

	// Create the maps for remapping
	cv::Mat UmapLeft, VmapLeft;
	cv::Mat UmapRight, VmapRight;
	int type = CV_32FC1; // Type of the map

	// Create the maps for the images
	cv::initUndistortRectifyMap(K1, d1, R1, P1, image1.size(), type, UmapLeft, VmapLeft);
	cv::initUndistortRectifyMap(K2, d2, R2, P2, image1.size(), type, UmapRight, VmapRight);

	// Remap the images
	cv::Mat image1Rectified, image2Rectified;
	cv::remap(image1, image1Rectified, UmapLeft, VmapLeft, cv::INTER_LINEAR);
	cv::remap(image2, image2Rectified, UmapRight, VmapRight, cv::INTER_LINEAR);

	// ===== Preprocess images =====
	// Resize the images to a smaller size for faster processing
	cv::Mat image1RS, image2RS;
	cv::resize(image1Rectified, image1RS, cv::Size(), 0.25, 0.25);
	cv::resize(image2Rectified, image2RS, cv::Size(), 0.25, 0.25);

	if (debugging) {
		std::cout << "Image 1 size before resize: " << std::endl << image1Rectified.size() << std::endl;
		std::cout << "Image 2 size before resize: " << std::endl << image2Rectified.size() << std::endl;
		std::cout << "Image 1 size after resize: " << std::endl << image1RS.size() << std::endl;
		std::cout << "Image 2 size after resize: " << std::endl << image2RS.size() << std::endl;
	}

	// Convert the images to grayscale
	cv::Mat grayLeft, grayRight;
	cv::cvtColor(image1RS, grayLeft, cv::COLOR_BGR2GRAY);
	cv::cvtColor(image2RS, grayRight, cv::COLOR_BGR2GRAY);

	// ===== Block Matching ======
	// Create StereoBM object
	int maxDisparity = 64; // Must be divisible by 16
	int blockSize = 21; // Must be odd
	cv::Ptr<cv::StereoBM> blockMatcher = cv::StereoBM::create(maxDisparity, blockSize);

	// Compute disparity map
	cv::Mat disparityBM;
	blockMatcher->compute(grayLeft, grayRight, disparityBM);

	// Display the images
	cv::namedWindow("Left", cv::WINDOW_NORMAL);
	cv::namedWindow("Right", cv::WINDOW_NORMAL);
	cv::namedWindow("Left_Remapped", cv::WINDOW_NORMAL);
	cv::namedWindow("Right_Remaped", cv::WINDOW_NORMAL);
	cv::imshow("Left", image1);
	cv::imshow("Right", image2);
	cv::imshow("Left_Remapped", image1RS);
	cv::imshow("Right_Remaped", image2RS);
	cv::waitKey();

	return 0;
}