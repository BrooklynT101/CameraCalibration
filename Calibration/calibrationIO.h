#pragma once

#include <opencv2/opencv.hpp>

void saveStereoCalibration(const std::string& filename, 
	const cv::Mat& K1, const cv::Mat& distCoeff1, 
	const cv::Mat& K2, const cv::Mat& distCoeff2, 
	const cv::Mat& R, const cv::Mat& t);

void readStereoCalibration(const std::string& filename,
	cv::Mat& K1, cv::Mat& distCoeff1, 
	cv::Mat& K2, cv::Mat& distCoeff2, 
	cv::Mat& R, cv::Mat& t);