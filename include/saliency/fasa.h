#pragma once
#include <iostream>
#include <time.h>
#include <dirent.h>
#include <fstream>

#include "opencv2/opencv.hpp"
using namespace cv;
using namespace std;

int get_saliency_map(cv::Mat &im, cv::Mat& saliency);
