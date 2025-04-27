#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <vector>
#include <iomanip> // For std::setfill and std::setw
#include <iostream>

static int displayCheckerBoardPattern() {
	// Setup OpenCV Windows, make them resizable
	cv::namedWindow("Left", cv::WINDOW_NORMAL);
	cv::namedWindow("Right", cv::WINDOW_NORMAL);

	const cv::Size patternSize(10, 5); // Number of internal corners per a chessboard row and column

	// Storage for detected corners for later calibration
	std::vector<std::vector<cv::Point2f>> allCornersLeft;
	std::vector<std::vector<cv::Point2f>> allCornersRight;
	// Store name of images where corners were not detected
	std::vector<std::string> failedImagesLeft;
	std::vector<std::string> failedImagesRight;

	for (int i = 457; i < 476; ++i) {

		// ----- Read in the left image ----
		std::ostringstream pathStreamL;
		pathStreamL << "data/CalibrationLeft/DSCF"
			<< std::setfill('0') << std::setw(4) << i
			<< "_L.JPG";
		std::string fnameL = pathStreamL.str(); // Converts the stream to a string

		cv::Mat imageL = cv::imread(fnameL);
		if (imageL.empty()) {
			std::cerr << "Could not open or find the image: " << fnameL << std::endl;
			continue;
		}


		// ----- Read right image ----
		std::ostringstream pathStreamR;
		pathStreamR << "data/CalibrationRight/DSCF"
			<< std::setfill('0') << std::setw(4) << i
			<< "_R.JPG";
		std::string fnameR = pathStreamR.str(); // Converts the stream to a string

		cv::Mat imageR = cv::imread(fnameR);
		if (imageR.empty()) {
			std::cerr << "Could not open or find the image: " << fnameR << std::endl;
			continue;
		}

		// --- Detect Checkerboard in Left Image ---
		std::vector<cv::Point2f> cornersL;
		bool foundL = cv::findChessboardCorners(imageL, patternSize, cornersL);

		if (foundL) {
			// Optional: refine corners
			/*cv::Mat grayL;
			cv::cvtColor(imageL, grayL, cv::COLOR_BGR2GRAY);
			cv::cornerSubPix(
				grayL, cornersL, cv::Size(11, 11), cv::Size(-1, -1),
				cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.001)
			);*/
			// Draw detected corners
			cv::drawChessboardCorners(imageL, patternSize, cornersL, foundL);
			allCornersLeft.push_back(cornersL);
		}
		else {
			std::cerr << "Checkerboard not found in left image: " << fnameL << std::endl;
			failedImagesLeft.push_back(fnameL);
			continue;
		}

		// ----- Detect Checkerboard corners in Right Image ----
		std::vector<cv::Point2f> cornersR;
		bool foundR = cv::findChessboardCorners(imageR, patternSize, cornersR);

		if (foundR) {
			// Refine the corner locations for calibration
			/*
			cv::Mat greyR;
			cv::cvtColor(imageR, greyR, cv::COLOR_BGR2GRAY);
			cv::cornerSubPix(greyR, cornersR, cv::Size(11, 11),
			cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.001));
			*/

			// Draw the corners on the image
			cv::drawChessboardCorners(imageR, patternSize, cornersR, foundR);

			allCornersRight.push_back(cornersR);
		}
		else {
			std::cerr << "Checkerboard not found in image: " << fnameR << std::endl;
			failedImagesRight.push_back(fnameR);
			continue;
		}

		// Display both images
		cv::imshow("Left", imageL);
		cv::imshow("Right", imageR);

		// Single wait for BOTH images
		int key = cv::waitKey(0);
		if (key == 27) { // ESC key pressed
			std::cout << "ESC pressed. Exiting early.\n";
			break;
		}
	}

	std::cout << "Finished displaying all calibration images.\n";
	std::cout << "Total images processed: " << allCornersLeft.size() + allCornersRight.size() << std::endl;
	if (allCornersLeft.size() != allCornersRight.size()) {
		std::cerr << "Warning: Number of detected corners in left and right images do not match!" << std::endl;

		std::cout << "Failed images (Left):\n";
		for (const auto& img : failedImagesLeft) {
			std::cout << img << std::endl;
		}
		std::cout << "Failed images (Right):\n";
		for (const auto& img : failedImagesRight) {
			std::cout << img << std::endl;
		}
	}

	return 0;
}

static int undistortImage(const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs) {
	// Load an image
	cv::Mat imageL = cv::imread("data/CalibrationLeft/DSCF0463_L.JPG");

	if (imageL.empty()) {
		std::cerr << "Error: Could not load the image for undistortion." << std::endl;
		return -1;
	}

	// Undistort the image
	cv::Mat undistorted;
	cv::undistort(imageL, undistorted, cameraMatrix, distCoeffs);

	// --- Enhancement 1: Draw guide lines on both images ---
	// Draw a horizontal center line
	cv::Mat imageWithLine = imageL.clone();
	cv::Mat undistortedWithLine = undistorted.clone();

	cv::line(imageWithLine, cv::Point(0, imageL.rows / 2), cv::Point(imageL.cols, imageL.rows / 2), cv::Scalar(0, 255, 0), 2);
	cv::line(undistortedWithLine, cv::Point(0, undistorted.rows / 2), cv::Point(undistorted.cols, undistorted.rows / 2), cv::Scalar(0, 255, 0), 2);

	// --- Enhancement 2: Create blended overlay comparison ---
	cv::Mat blended;
	cv::addWeighted(imageL, 0.5, undistorted, 0.5, 0, blended);

	// --- Display each view separately ---
	cv::namedWindow("Original with Line", cv::WINDOW_NORMAL);
	cv::namedWindow("Undistorted with Line", cv::WINDOW_NORMAL);
	cv::namedWindow("Blended Overlay Comparison", cv::WINDOW_NORMAL);

	cv::imshow("Original with Line", imageWithLine);
	cv::imshow("Undistorted with Line", undistortedWithLine);
	cv::imshow("Blended Overlay Comparison", blended);

	cv::waitKey(0);
	return 0;
}

static int calibrateBothSets() {
	// --- debugging flag ---
	const bool useGrayscalePreprocessing = true;

	// --- Setup calibration pattern info ---
	const cv::Size patternSize(10, 5); // 10 internal corners wide, 5 tall
	const float squareSize = 47.0f;    // 47mm per square
	const cv::Size imageSize(1920, 1080); // Image size (in pixels)

	// --- Storage for all detections ---
	std::vector<std::vector<cv::Point3f>> objectPointsLeft, objectPointsRight; // 3D points in world space
	std::vector<std::vector<cv::Point2f>> imagePointsLeft, imagePointsRight;  // 2D points in image space

	// --- Build the checkerboard model points (real-world 3D) ---
	std::vector<cv::Point3f> checkerboardPattern;
	for (int y = 0; y < patternSize.height; ++y) {
		for (int x = 0; x < patternSize.width; ++x) {
			checkerboardPattern.push_back(cv::Point3f(x * squareSize, y * squareSize, 0.0f));
		}
	}

	// Loop over all images
	for (int i = 457; i < 476; ++i) {
		// --- Load Left Image ---
		std::ostringstream pathStreamL;
		pathStreamL << "data/CalibrationLeft/DSCF"
			<< std::setfill('0') << std::setw(4) << i
			<< "_L.JPG";
		std::string fnameL = pathStreamL.str();
		cv::Mat imageL = cv::imread(fnameL);
		if (imageL.empty()) {
			std::cerr << "Could not open left image: " << fnameL << std::endl;
			continue;
		}

		// --- Load Right Image ---
		std::ostringstream pathStreamR;
		pathStreamR << "data/CalibrationRight/DSCF"
			<< std::setfill('0') << std::setw(4) << i
			<< "_R.JPG";
		std::string fnameR = pathStreamR.str();
		cv::Mat imageR = cv::imread(fnameR);
		if (imageR.empty()) {
			std::cerr << "Could not open right image: " << fnameR << std::endl;
			continue;
		}

		// --- Prepare images for detection ---
		cv::Mat detectImageL = imageL;
		cv::Mat detectImageR = imageR;
		int flags = 0;

		if (useGrayscalePreprocessing) {
			cv::cvtColor(imageL, detectImageL, cv::COLOR_BGR2GRAY);
			cv::cvtColor(imageR, detectImageR, cv::COLOR_BGR2GRAY);
			flags = cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK;
		}

		// --- Detect corners in Both Images ---
		std::vector<cv::Point2f> cornersL, cornersR;
		bool foundL = cv::findChessboardCorners(detectImageL, patternSize, cornersL, flags);
		bool foundR = cv::findChessboardCorners(detectImageR, patternSize, cornersR, flags);

		// Only save if **both** detections succeed
		if (foundL && foundR) {
			// Optional refinement (always done on grayscale)
			cv::Mat grayL, grayR;
			if (!useGrayscalePreprocessing) {
				cv::cvtColor(imageL, grayL, cv::COLOR_BGR2GRAY);
				cv::cvtColor(imageR, grayR, cv::COLOR_BGR2GRAY);
			}
			else {
				grayL = detectImageL;
				grayR = detectImageR;
			}

			cv::cornerSubPix(
				grayL, cornersL, cv::Size(11, 11), cv::Size(-1, -1),
				cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.001)
			);
			cv::cornerSubPix(
				grayR, cornersR, cv::Size(11, 11), cv::Size(-1, -1),
				cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.001)
			);

			imagePointsLeft.push_back(cornersL);
			objectPointsLeft.push_back(checkerboardPattern);

			imagePointsRight.push_back(cornersR);
			objectPointsRight.push_back(checkerboardPattern);
		}
		else {
			std::cout << "Checkerboard detection failed for pair: " << fnameL << " and " << fnameR << std::endl;
		}
	}

	// --- Now calibrate Left Camera ---
	if (imagePointsLeft.empty()) {
		std::cerr << "No valid detections for Left camera. Calibration failed!" << std::endl;
		return -1;
	}

	cv::Mat cameraMatrixLeft, distCoeffsLeft;
	std::vector<cv::Mat> rvecsLeft, tvecsLeft;

	double reprojectionErrorLeft = cv::calibrateCamera(
		objectPointsLeft, imagePointsLeft, imageSize,
		cameraMatrixLeft, distCoeffsLeft,
		rvecsLeft, tvecsLeft
	);

	// --- Now calibrate Right Camera ---
	if (imagePointsRight.empty()) {
		std::cerr << "No valid detections for Right camera. Calibration failed!" << std::endl;
		return -1;
	}

	cv::Mat cameraMatrixRight, distCoeffsRight;
	std::vector<cv::Mat> rvecsRight, tvecsRight;

	double reprojectionErrorRight = cv::calibrateCamera(
		objectPointsRight, imagePointsRight, imageSize,
		cameraMatrixRight, distCoeffsRight,
		rvecsRight, tvecsRight
	);

	// --- Save Results ---
	cv::FileStorage fsLeft("left_camera_calibration.yml", cv::FileStorage::WRITE);
	fsLeft << "CameraMatrix" << cameraMatrixLeft;
	fsLeft << "DistCoeffs" << distCoeffsLeft;
	fsLeft.release();

	cv::FileStorage fsRight("right_camera_calibration.yml", cv::FileStorage::WRITE);
	fsRight << "CameraMatrix" << cameraMatrixRight;
	fsRight << "DistCoeffs" << distCoeffsRight;
	fsRight.release();

	// --- Print Results ---
	std::cout << "\n=== Left Camera Calibration ===\n";
	std::cout << "Reprojection Error = " << reprojectionErrorLeft << " pixels\n";
	std::cout << "Camera Matrix (K):\n" << cameraMatrixLeft << "\n";
	std::cout << "Distortion Coefficients:\n" << distCoeffsLeft << "\n";

	std::cout << "\n=== Right Camera Calibration ===\n";
	std::cout << "Reprojection Error = " << reprojectionErrorRight << " pixels\n";
	std::cout << "Camera Matrix (K):\n" << cameraMatrixRight << "\n";
	std::cout << "Distortion Coefficients:\n" << distCoeffsRight << "\n";

	return 0;
}

static int testStereoDifference() {
	// Load the left and right images
	cv::Mat leftImage = cv::imread("data/CalibrationLeft/DSCF0455_L.JPG");
	cv::Mat rightImage = cv::imread("data/CalibrationRight/DSCF0455_R.JPG");

	if (leftImage.empty() || rightImage.empty()) {
		std::cerr << "Error: Could not load one or both images for stereo difference test." << std::endl;
		return -1;
	}

	//// Convert to grayscale
	//cv::Mat leftGray, rightGray;
	//cv::cvtColor(leftImage, leftGray, cv::COLOR_BGR2GRAY);
	//cv::cvtColor(rightImage, rightGray, cv::COLOR_BGR2GRAY);

	// Resize images if they aren't exactly the same size (sometimes minor size diff happens)
	if (leftImage.size() != rightImage.size()) {
		std::cerr << "Warning: Resizing images to match." << std::endl;
		cv::resize(rightImage, rightImage, leftImage.size());
	}

	// Compute absolute difference
	cv::Mat diff;
	cv::absdiff(leftImage, rightImage, diff);

	// Calculate a simple metric: mean pixel difference
	cv::Scalar meanDiff = cv::mean(diff);

	std::cout << "Average pixel difference between Left and Right images: " << meanDiff[0] << std::endl;

	// Create a blended overlay comparison 
	cv::Mat blended;
	cv::addWeighted(leftImage, 0.5, rightImage, 0.5, 0, blended);

	// Display difference visually
	cv::namedWindow("Difference", cv::WINDOW_NORMAL);
	cv::namedWindow("Blended Overlay", cv::WINDOW_NORMAL);
	cv::imshow("Difference", diff);
	cv::imshow("Blended Overlay", blended);
	cv::waitKey(0);

	return 0;
}

static int stereoCalibratePair() {
	// ----- Setup calibration pattern info -----
	const cv::Size patternSize(10, 5); // Checkerboard internal corners
	const float squareSize = 47.0f;    // Square size in mm
	const cv::Size imageSize(1920, 1080); // Calibration image size
	const bool refineCorners = false; // Refine corner locations

	// ----- Load the camera matrices and distortion coefficients -----
	// Load the left camera calibration data
	cv::Mat cameraMatrixLeft, distCoeffsLeft;
	{
		cv::FileStorage fsLeft("left_camera_calibration.yml", cv::FileStorage::READ);
		if (!fsLeft.isOpened()) {
			std::cerr << "Error: Could not open left_camera_calibration.yml\n";
			return -1;
		}
		fsLeft["CameraMatrix"] >> cameraMatrixLeft;
		fsLeft["DistCoeffs"] >> distCoeffsLeft;
		fsLeft.release();
	}

	// Load the right camera calibration data
	cv::Mat cameraMatrixRight, distCoeffsRight;
	{
		cv::FileStorage fsRight("right_camera_calibration.yml", cv::FileStorage::READ);
		if (!fsRight.isOpened()) {
			std::cerr << "Error: Could not open right_camera_calibration.yml\n";
			return -1;
		}
		fsRight["CameraMatrix"] >> cameraMatrixRight;
		fsRight["DistCoeffs"] >> distCoeffsRight;
		fsRight.release();
	}

	std::cout << "\n=== Loaded Camera Matrices Before Stereo Calibration ===\n";
	std::cout << "Left Camera Matrix (Before):\n" << cameraMatrixLeft << "\n";
	std::cout << "Right Camera Matrix (Before):\n" << cameraMatrixRight << "\n";

	// ----- Prepare object points and image points -----
	std::vector<std::vector<cv::Point3f>> objectPoints;
	std::vector<std::vector<cv::Point2f>> imagePointsLeft, imagePointsRight;

	// Build the checkerboard model points (real-world 3D)
	std::vector<cv::Point3f> checkerboardPattern;
	for (int y = 0; y < patternSize.height; ++y) {
		for (int x = 0; x < patternSize.width; ++x) {
			checkerboardPattern.push_back(cv::Point3f(x * squareSize, y * squareSize, 0.0f));
		}
	}

	// ----- Detect corners in all pairs -----
	int validPairs = 0;
	for (int i = 457; i < 476; ++i) {
		// Load left image
		std::ostringstream pathStreamL;
		pathStreamL << "data/CalibrationLeft/DSCF"
			<< std::setfill('0') << std::setw(4) << i
			<< "_L.JPG";
		std::string fnameL = pathStreamL.str();
		cv::Mat imageL = cv::imread(fnameL);
		if (imageL.empty()) {
			std::cerr << "Could not open left image: " << fnameL << std::endl;
			continue;
		}
		// Load right image
		std::ostringstream pathStreamR;
		pathStreamR << "data/CalibrationRight/DSCF"
			<< std::setfill('0') << std::setw(4) << i
			<< "_R.JPG";
		std::string fnameR = pathStreamR.str();
		cv::Mat imageR = cv::imread(fnameR);
		if (imageR.empty()) {
			std::cerr << "Could not open right image: " << fnameR << std::endl;
			continue;
		}

		//Find Corners
		std::vector<cv::Point2f> cornersL, cornersR;
		bool foundL = cv::findChessboardCorners(imageL, patternSize, cornersL);
		bool foundR = cv::findChessboardCorners(imageR, patternSize, cornersR);

		// Only save if **both** detections succeed
		if (foundL && foundR) {
			// Refine corners optional
			if (refineCorners) {
				cv::Mat grayL, grayR;
				cv::cvtColor(imageL, grayL, cv::COLOR_BGR2GRAY);
				cv::cvtColor(imageR, grayR, cv::COLOR_BGR2GRAY);
				cv::cornerSubPix(grayL, cornersL, cv::Size(11, 11), cv::Size(-1, -1),
					cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.001));
				cv::cornerSubPix(grayR, cornersR, cv::Size(11, 11), cv::Size(-1, -1),
					cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.001));
			}

			// Draw detected corners
			imagePointsLeft.push_back(cornersL);
			imagePointsRight.push_back(cornersR);
			objectPoints.push_back(checkerboardPattern);
			++validPairs;
		}
		else {
			std::cout << "Checkerboard detection failed for pair: " << fnameL << " and " << fnameR << std::endl;
		}
	}

	std::cout << "Total valid stereo pairs: " << validPairs << "\n";

	// --- Check if there are enough valid detections ---
	if (objectPoints.size() < 5) { // Arbitrary low threshold
		std::cerr << "Not enough valid checkerboard detections for reliable stereo calibration.\n";
		return -1;
	}

	// --- Stereo Calibration ---
	if (objectPoints.empty()) {
		std::cerr << "No valid checkerboard detections found for stereo calibration.\n";
		return -1;
	}

	cv::Mat R, T, E, F;

	double stereoError = cv::stereoCalibrate(
		objectPoints,
		imagePointsLeft,
		imagePointsRight,
		cameraMatrixLeft,
		distCoeffsLeft,
		cameraMatrixRight,
		distCoeffsRight,
		imageSize,
		R, T, E, F
		//cv::CALIB_FIX_INTRINSIC | cv::CALIB_USE_INTRINSIC_GUESS// <-- Important: Keep intrinsics fixed during stereo calibration!
	);

	std::cout << "\n=== Camera Matrices After Stereo Calibration ===\n";
	std::cout << "Left Camera Matrix (After):\n" << cameraMatrixLeft << "\n";
	std::cout << "Right Camera Matrix (After):\n" << cameraMatrixRight << "\n";


	// --- Save Stereo Calibration Results ---
	cv::FileStorage fs("stereo_calibration.yml", cv::FileStorage::WRITE);
	if (!fs.isOpened()) {
		std::cerr << "Error: Could not open stereo_calibration.yml for writing\n";
		return -1;
	}
	fs << "RotationMatrix" << R;
	fs << "TranslationVector" << T;
	fs << "EssentialMatrix" << E;
	fs << "FundamentalMatrix" << F;
	fs.release();

	// --- Print Results ---
	std::cout << "\n=== Stereo Calibration ===\n";
	std::cout << "Stereo Reprojection Error = " << stereoError << " pixels\n";
	std::cout << "Rotation Matrix (R):\n" << R << "\n";
	std::cout << "Translation Vector (T):\n" << T << "\n";

	return 0;
}

#include <random> // For random number generation

static int stereoRectifyAndDisplay() {
	// --- Load Stereo Calibration Results ---
	cv::Mat cameraMatrixLeft, distCoeffsLeft;
	cv::Mat cameraMatrixRight, distCoeffsRight;
	cv::Mat R, T;

	{
		cv::FileStorage fsLeft("left_camera_calibration.yml", cv::FileStorage::READ);
		if (!fsLeft.isOpened()) {
			std::cerr << "Error: Could not open left_camera_calibration.yml\n";
			return -1;
		}
		fsLeft["CameraMatrix"] >> cameraMatrixLeft;
		fsLeft["DistCoeffs"] >> distCoeffsLeft;
		fsLeft.release();
	}
	{
		cv::FileStorage fsRight("right_camera_calibration.yml", cv::FileStorage::READ);
		if (!fsRight.isOpened()) {
			std::cerr << "Error: Could not open right_camera_calibration.yml\n";
			return -1;
		}
		fsRight["CameraMatrix"] >> cameraMatrixRight;
		fsRight["DistCoeffs"] >> distCoeffsRight;
		fsRight.release();
	}
	{
		cv::FileStorage fsStereo("stereo_calibration.yml", cv::FileStorage::READ);
		if (!fsStereo.isOpened()) {
			std::cerr << "Error: Could not open stereo_calibration.yml\n";
			return -1;
		}
		fsStereo["RotationMatrix"] >> R;
		fsStereo["TranslationVector"] >> T;
		fsStereo.release();
	}

	const cv::Size imageSize(1920, 1080);

	// --- Stereo Rectify ---
	cv::Mat R1, R2, P1, P2, Q;
	cv::Rect validRoi1, validRoi2;

	cv::stereoRectify(
		cameraMatrixLeft, distCoeffsLeft,
		cameraMatrixRight, distCoeffsRight,
		imageSize, R, T,
		R1, R2, P1, P2, Q,
		cv::CALIB_ZERO_DISPARITY, 1.0, imageSize, &validRoi1, &validRoi2
	);

	// --- Create Rectification Maps ---
	cv::Mat mapLx, mapLy, mapRx, mapRy;
	cv::initUndistortRectifyMap(
		cameraMatrixLeft, distCoeffsLeft, R1, P1, imageSize, CV_32FC1, mapLx, mapLy
	);
	cv::initUndistortRectifyMap(
		cameraMatrixRight, distCoeffsRight, R2, P2, imageSize, CV_32FC1, mapRx, mapRy
	);

	// --- Pick a Random Image Index ---
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> distrib(457, 475);
	int randomIndex = distrib(gen);

	// --- Load the Left and Right Images ---
	std::ostringstream pathStreamL, pathStreamR;
	pathStreamL << "data/CalibrationLeft/DSCF"
		<< std::setfill('0') << std::setw(4) << randomIndex
		<< "_L.JPG";
	std::string fnameL = pathStreamL.str();

	pathStreamR << "data/CalibrationRight/DSCF"
		<< std::setfill('0') << std::setw(4) << randomIndex
		<< "_R.JPG";
	std::string fnameR = pathStreamR.str();

	cv::Mat imageL = cv::imread(fnameL);
	cv::Mat imageR = cv::imread(fnameR);

	if (imageL.empty() || imageR.empty()) {
		std::cerr << "Error: Could not load one or both images.\n";
		return -1;
	}

	// --- Apply Rectification ---
	cv::Mat rectifiedL, rectifiedR;
	cv::remap(imageL, rectifiedL, mapLx, mapLy, cv::INTER_LINEAR);
	cv::remap(imageR, rectifiedR, mapRx, mapRy, cv::INTER_LINEAR);

	// --- Display Results ---
	cv::namedWindow("Original Left", cv::WINDOW_NORMAL);
	cv::namedWindow("Original Right", cv::WINDOW_NORMAL);
	cv::namedWindow("Rectified Left", cv::WINDOW_NORMAL);
	cv::namedWindow("Rectified Right", cv::WINDOW_NORMAL);
	cv::namedWindow("Rectified Left with Lines", cv::WINDOW_NORMAL);
	cv::namedWindow("Rectified Right with Lines", cv::WINDOW_NORMAL);
	cv::namedWindow("Original Left with lines", cv::WINDOW_NORMAL);
	cv::namedWindow("Original Right with lines", cv::WINDOW_NORMAL);

	cv::imshow("Original Left", imageL);
	cv::imshow("Original Right", imageR);
	cv::imshow("Rectified Left", rectifiedL);
	cv::imshow("Rectified Right", rectifiedR);

	// --- Add Optional Horizontal Lines for Visualizing Rectification ---
	for (int y = 0; y < rectifiedL.rows; y += 50) {
		cv::line(rectifiedL, cv::Point(0, y), cv::Point(rectifiedL.cols, y), cv::Scalar(0, 255, 0), 1);
		cv::line(rectifiedR, cv::Point(0, y), cv::Point(rectifiedR.cols, y), cv::Scalar(0, 255, 0), 1);
		cv::line(imageL, cv::Point(0, y), cv::Point(rectifiedL.cols, y), cv::Scalar(0, 255, 0), 1);
		cv::line(imageR, cv::Point(0, y), cv::Point(rectifiedL.cols, y), cv::Scalar(0, 255, 0), 1);
	}

	cv::imshow("Rectified Left with Lines", rectifiedL);
	cv::imshow("Rectified Right with Lines", rectifiedR);
	cv::imshow("Original Left with lines", imageL);
	cv::imshow("Original Right with lines", imageR);

	std::cout << "Displayed stereo pair #" << randomIndex << " rectified.\n";

	cv::waitKey(0);
	return 0;
}


// Main function to run the processes
int main(int argc, char* argv[]) {
	//testStereoDifference();
	//displayCheckerBoardPattern();
	//calibrateBothSets();
	//stereoCalibratePair();
	stereoRectifyAndDisplay();
	return 0;
}

