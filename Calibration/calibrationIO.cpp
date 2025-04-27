#include "calibrationIO.h"

void saveStereoCalibration(const std::string& filename,
	const cv::Mat& K1, const cv::Mat& distCoeff1,
	const cv::Mat& K2, const cv::Mat& distCoeff2,
	const cv::Mat& R, const cv::Mat& t) {
	
	cv::FileStorage fsw(filename, cv::FileStorage::WRITE);

	fsw << "K1" << K1;
	fsw << "d1" << distCoeff1;
	fsw << "K2" << K2;
	fsw << "d2" << distCoeff2;
	fsw << "R" << R;
	fsw << "t" << t;

	fsw.release();

}

void readStereoCalibration(const std::string& filename,
	cv::Mat& K1, cv::Mat& distCoeff1,
	cv::Mat& K2, cv::Mat& distCoeff2,
	cv::Mat& R, cv::Mat& t) {

	cv::FileStorage fsr(filename, cv::FileStorage::READ);
	
	fsr["K1"] >> K1;
	fsr["d1"] >> distCoeff1;
	fsr["K2"] >> K2;
	fsr["d2"] >> distCoeff2;
	fsr["R"] >> R;
	fsr["t"] >> t;

	fsr.release();
}