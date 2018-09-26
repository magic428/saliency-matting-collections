#pragma once


#include "common/my_common.h"

enum{CV_FLIP_BOTH = -1, CV_FLIP_VERTICAL = 0, CV_FLIP_HORIZONTAL = 1};

struct CmCv {
	// AbsAngle: Calculate magnitude and angle of vectors.
    static void AbsAngle(cv::Mat& cmplx32FC2, cv::Mat& mag32FC1, cv::Mat& ang32FC1);

	// GetCmplx: Get a complex value image from it's magnitude and angle.
    static void GetCmplx(cv::Mat& mag32F, cv::Mat& ang32F, cv::Mat& cmplx32FC2);

	// Mat2GrayLog: Convert and arbitrary mat to [0, 1] for display.
	// The result image is in 32FCn format and range [0, 1.0].
	// Mat2GrayLinear(log(img+1), newImg). In place operation is supported.
    static void Mat2GrayLog(cv::Mat& img, cv::Mat& newImg);

	// Low frequency part is always been move to the central part:
	//				 -------                          -------	
	//				| 1 | 2 |                        | 3 | 4 |	
	//				 -------            -->           -------	
	//				| 4 | 3 |                        | 2 | 1 |	
	//				 -------                          -------	
    static void FFTShift(cv::Mat& img);

    // Swap the content of two cv::Mat with same type and size
    static inline void Swap(cv::Mat& a, cv::Mat& b);

	// Normalize size/image to min(width, height) = shortLen and use width 
	// and height to be multiples of unitLen while keeping its aspect ratio 
	// as much as possible. unitLen must not be 0.
	static inline Size NormalizeSize(const Size& sz, int shortLen, int unitLen = 1);
    static inline void NormalizeImg(cv::Mat&img, cv::Mat& dstImg, int shortLen = 256, int unitLen = 8);
	static void NormalizeImg(CStr &inDir, CStr &outDir, int minLen = 300, bool subFolders = true);

	// Get image region by two corner point.
	static inline Rect GetImgRange(Point p1, Point p2, Size imgSz);

	// Check an image (with size imgSz) point and correct it if necessary
	static inline void CheckPoint(Point &p, Size imgSz);

    static inline cv::Mat Merge(cv::Mat &m3c, cv::Mat &m1c); // Merge a 3 channel and 1 channel mat to 4 channel one
    static inline void Split(cv::Mat &m4c, cv::Mat &m3c, cv::Mat &m1c);
	
	// Get mask region. 
	static Rect GetMaskRange(cv::Mat &mask1u, int ext = 0, int thresh = 10);
	
	// Get continuous components for same label regions. Return region index mat,
	// index-counter pair (Number of pixels for each index), and label of each idx
	static int GetRegions(const Mat_<byte> &label1u, Mat_<int> &regIdx1i, vecI &idxCount, vecB &idxLabel, bool noZero = false);
	static int GetRegions(const Mat_<byte> &label1u, Mat_<int> &regIdx1i, vecI &idxCount, bool noZero = false) {vecB idxLabel; return GetRegions(label1u, regIdx1i, idxCount, idxLabel, noZero);}

	// Get continuous components for non-zero labels. Return region index mat (region index 
	// of each mat position) and sum of label values in each region
	static int GetNZRegions(const Mat_<byte> &label1u, Mat_<int> &regIdx1i, vecI &idxSum);

	// Get continuous None-Zero label Region with Largest Sum value
    static cv::Mat GetNZRegionsLS(cv::Mat &mask1u, double ignoreRatio = 0.02);
	
	// Get points in border regions
	static int GetBorderPnts(Size sz, double ratio, vector<Point> &bdPnts);

	// Get border regions, which typically corresponds to background region
    static cv::Mat GetBorderReg(cv::Mat &idx1i, int regNum, double ratio = 0.02, double thr = 0.4);
    static cv::Mat GetBorderRegC(cv::Mat &img3u, cv::Mat &idx1i, vecI &idxCount);

    static void fillPoly(cv::Mat& img, const vector<PointSeti> _pnts, const Scalar& color, int lineType = 8, int shift = 0, Point offset = Point());

	// down sample without convolution, similar to cv::pyrDown
    template<class T> static void PyrDownSample(cv::Mat &src, cv::Mat &dst);
    template<class T> static void PyrUpSample(cv::Mat &src, cv::Mat &dst, Size dSz);

	static void  SaveImgRGB(CStr &fName, cv::Mat &img);// Saving RGB image for QImage data

	static void Demo(const char* fileName = "H:\\Resize\\cd3.avi");

	//// Adding alpha value to img to show. img: 8U3C, alpha 8U1C
    static void AddAlpha(cv::Mat &fg3u, cv::Mat &alpha1u, cv::Mat &res3u);
    static void AddAlpha(cv::Mat &bg3u, cv::Mat &fg3u, cv::Mat &alpha1u, cv::Mat &res3u);
	//static void AddAlpha(CvMat *img, CvMat *alpha, CvScalar bgColor);

    static inline cv::Mat getContinouse(cv::Mat &mat) {cv::Mat tmp; mat.copyTo(tmp); return tmp; }


	// Average multi-channel float values within each region. 
	// Region index should be int values in range [0, regNum -1]
    static void avgPerRegion(cv::Mat &regIdx1i, cv::Mat &unaryNf, int regNum);

    template <typename T> static cv::Mat addChannel(cv::Mat &mat, int num = 1, double defaultVal = 0);

    static void CannySimpleRGB(cv::Mat &img3u, cv::Mat &edge1u, double thresh1, double thresh2, int apertureSize, bool L2gradient = false);
    static cv::Mat getGrabMask(cv::Mat &img3u, Rect rect);//, CStr sameNameNE, int ext = 4
    static void rubustifyBorderMask(cv::Mat& mask1u);

	static int intMatMax(cv::Mat idx1i); // return the max value in an int matrix
};

