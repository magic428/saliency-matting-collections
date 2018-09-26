#pragma once

#include "common/CmLib.h"

struct CmSaliencyRC
{
    typedef cv::Mat (*GET_SAL_FUNC)(cv::Mat &);

	// Get saliency values of a group of images.
	// Input image names and directory name for saving saliency maps.
	static void Get(std::string &imgNameW, std::string &salDir);

	// Evaluate saliency detection methods. Input ground truth file names and saliency map directory
	static void Evaluate(std::string gtW, std::string &salDir, std::string &resName);

	// Frequency Tuned [1].
    static cv::Mat GetFT(cv::Mat &img3f);

	// Histogram Contrast of [3]
    static cv::Mat GetHC(cv::Mat &img3f);

	// Region Contrast 
    static cv::Mat GetRC(cv::Mat &img3f);
    static cv::Mat GetRC(cv::Mat &img3f, cv::Mat &idx1i, int regNum, double sigmaDist = 0.4);
    static cv::Mat GetRC(cv::Mat &img3f, double sigmaDist, double segK, int segMinSize, double segSigma);

	// Luminance Contrast [2]
    static cv::Mat GetLC(cv::Mat &img3f);

	// Spectral Residual [4]
    static cv::Mat GetSR(cv::Mat &img3f);


    static cv::Mat Get(cv::Mat &img3f, GET_SAL_FUNC fun, int wkSize);
    static void SmoothByHist(cv::Mat &img3f, cv::Mat &sal1f, float delta);
    static void SmoothByRegion(cv::Mat &sal1f, cv::Mat &idx1i, int regNum, bool bNormalize = true);
    static void SmoothByGMMs(cv::Mat &img3f, cv::Mat &sal1f, int fNum = 5, int bNum = 5, int wkSize = 0);

	static int Demo(std::string wkDir);

private:
	static const int SAL_TYPE_NUM = 5; 
	static const char* SAL_TYPE_DES[SAL_TYPE_NUM];
	static const GET_SAL_FUNC gFuns[SAL_TYPE_NUM];

	// Histogram based Contrast
    static void GetHC(cv::Mat &binColor3f, cv::Mat &colorNums1i, cv::Mat &colorSaliency);

    static void SmoothSaliency(cv::Mat &sal1f, float delta, const vector<vector<CostfIdx>> &similar);
    static void SmoothSaliency(cv::Mat &colorNum1i, cv::Mat &sal1f, float delta, const vector<vector<CostfIdx>> &similar);

	struct Region{
		Region() { pixNum = 0; ad2c = Point2d(0, 0);}
		int pixNum;  // Number of pixels
		vector<CostfIdx> freIdx;  // Frequency of each color and its index
		Point2d centroid;
		Point2d ad2c; // Average distance to image center
	};
	static void BuildRegions(cv::Mat& regIdx1i, vector<Region> &regs, cv::Mat &colorIdx1i, int colorNum);
    static void RegionContrast(const vector<Region> &regs, cv::Mat &color3fv, cv::Mat& regSal1d, double sigmaDist);

    static int Quantize(cv::Mat& img3f, cv::Mat &idx1i, cv::Mat &_color3f, cv::Mat &_colorNum, double ratio = 0.95, const int colorNums[3] = DefaultNums);
	static const int DefaultNums[3];

	// Get border regions, which typically corresponds to background region
    static cv::Mat GetBorderReg(cv::Mat &idx1i, int regNum, double ratio = 0.02, double thr = 0.3);

	// AbsAngle: Calculate magnitude and angle of vectors.
    static void AbsAngle(cv::Mat& cmplx32FC2, cv::Mat& mag32FC1, cv::Mat& ang32FC1);

	// GetCmplx: Get a complex value image from it's magnitude and angle.
    static void GetCmplx(cv::Mat& mag32F, cv::Mat& ang32F, cv::Mat& cmplx32FC2);
};

/************************************************************************/
/*[1]R. Achanta, S. Hemami, F. Estrada and S. Susstrunk, Frequency-tuned*/
/*   Salient Region Detection, IEEE CVPR, 2009.							*/
/*[2]Y. Zhai and M. Shah. Visual attention detection in video sequences */
/*   using spatiotemporal cues. In ACM Multimedia 2006.					*/
/*[3]M.-M. Cheng, N. J. Mitra, X. Huang, P.H.S. Torr S.-M. Hu. Global	*/
/*   Contrast based Salient Region Detection. IEEE PAMI, 2014.			*/
/*[4]X. Hou and L. Zhang. Saliency detection: A spectral residual		*/
/*   approach. In IEEE CVPR 2007, 2007.									*/
/************************************************************************/
