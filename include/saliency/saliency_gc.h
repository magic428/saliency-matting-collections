#pragma once

/************************************************************************/
/* For educational and research use only; commercial use are forbidden.	*/
/*
/* Download more source code from: http://mmcheng.net/					*/
/* If you use any part of the source code, please cite related papers:	*/
/* [1] SalientShape: Group Saliency in Image Collections. M.M. Cheng,	*/
/*	 N.J. Mitra, X. Huang, S.M. Hu. The Visual Computer, 2013.			*/
/* [2] Efficient Salient Region Detection with Soft Image Abstraction.	*/
/*	 M.M. Cheng, J. Warrell, W.Y. Lin, S. Zheng, V. Vineet, N. Crook.	*/
/*	 IEEE ICCV, 2013.													*/
/* [3] Salient Object Detection and Segmentation. M.M. Cheng, N.J.		*/
/*   Mitra, X. Huang, P.H.S. Torr, S.M. Hu. Submitted to IEEE TPAMI		*/
/*	 (TPAMI-2011-10-0753), 2011.										*/
/* [4] Global Contrast based Salient Region Detection, Cheng et. al.,	*/
/*	   CVPR 2015.														*/
/*	   														            */
/************************************************************************/
/* See Section 3.3 in the paper for the 4 layers representation:        */
/*  L0: Pixel layer, allow full resolution result, typical size 400x300.*/
/*	L1: Histogram layer, typical size 85, see reference [3] above.		*/
/*  L2: GMM layer, typical size 15.										*/
/*	L3: Clustered layer: the size is even smaller than layer 2.			*/
/************************************************************************/

#include "common/CmLib.h"

class CmSaliencyGC
{
public:
    // If not get CSD and only calculating Global uniqueness, the program will be faster
    CmSaliencyGC(CMat &img3f, CStr &outName, bool GET_CSD = true);

    void HistgramGMMs();
    void MergeGMMs();
    cv::Mat GetSaliencyCues();

    static int Demo(CStr imgW, CStr salDir); // C:/Data/SaliencyFT/Imgs/*.jpg  C:/Data/SaliencyFT/Saliency/

private: // Data values
    CmGMM _gmm;
    cv::Mat _img3f;
    CStr _nameNE;
    static const int DEFAULT_GMM_NUM;

    // L0: pixel layer, each cv::Mat variable has the same size as image
    cv::Mat _HistBinIdx1i; // The histogram bin index of each pixel, associating L0 and L1
    vector<cv::Mat> _PixelSalCi1f; // Pixel saliency of component i, type CV_32FC1 (denoted 1f)

    // L1: histogram layer, each cv::Mat variable has the same size as histogram
    vector<cv::Mat> _HistBinSalCi1f; // Histogram bin saliency of component i

    // L2: GMM layer, each vector has the size of GMM numbers
    int _NUM;
    vector<Vec3f> _gmmClrs; // Colors of GMM components
    vecD _gmmW; // Weight of each GMM component
    PointSetd _gmmMeanPos;
    vecI _ClusteredIdx; // Bridge between L2 and L3
    vecD _csd; // color spatial distribution
    vecD _gu; // global contrast
    vecD _fSal; // Final saliency

    // L3: clustered layer
    int _NUM_Merged;
    vector<cv::Mat> _pciM1f; // Probability of GMM components, after merged

    const bool _GET_CSD;

private:
    void ViewValues(vecD &vals, CStr &ext);
    cv::Mat GetSalFromGMMs(vecD &val, bool normlize = true);
    void GetCSD(vecD &csd, vecD &cD);
    void GetGU(vecD& gc, vecD &d, double sigmaDist, double dominate = 30);
    void ReScale(vecD& salVals, double minRatio = 0.01);

    // Return the spatial variance. Pixel locations are normalized to [0, 1]
    static double SpatialVar(CMat& map1f, double &cD = dummyD);
};
