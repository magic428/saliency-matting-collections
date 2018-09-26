#pragma once
/************************************************************************/
/* SaleicyCut: Unsupervised salient object segmentation     */
/*[1] MM Cheng, NJ Mitra, X Huang, PHS Torr SM Hu. Global Contrast  */
/*   based Salient Region Detection. IEEE CVPR, p. 409-416, 2011.  */
/************************************************************************/



// #include "common/my_common.h"
#include "cluster/gmm.h"
#include "segmentation/Maxflow/graph.h"
#include "common/CmCv.h"
#include "common/file_ops.h"
#include "saliency/fasa.h"

class CmSalCut
{
public: // Functions for saliency cut
    // User supplied Trimap values
    enum TrimapValue {TrimapBackground = 0, TrimapUnknown = 128, TrimapForeground = 255};

    CmSalCut(cv::Mat &img3f);
    ~CmSalCut(void);

    // Refer initialize for parameters
    static cv::Mat CutObjs( cv::Mat &img3f, cv::Mat &sal1f, cv::Mat &borderMask, 
        float t1 = 0.2f, float t2 = 0.9f, int wkSize = 20);
        
    // TODO: ref error
    // static cv::Mat CutObjs(cv::Mat &img3f, cv::Mat &sal1f, float t1 = 0.2f, float t2 = 0.9f,
        // cv::Mat &borderMask = cv::Mat(), int wkSize = 20);

    static int Demo(std::string imgNameW, std::string salDir);

public: // Functions for GrabCut

    // Initial rect region in between thr1 and thr2 and others below thr1 as the Grabcut paper 
    void initialize(const Rect &rect); 

    // Initialize using saliency map. In the Trimap: background < t1, foreground > t2, others unknown.
    // Saliency values are in [0, 1], "sal1f" and "1-sal1f" are used as weight to train fore and back ground GMMs
    void initialize(cv::Mat &sal1f, float t1, float t2);
    void initialize(cv::Mat &sal1u); // Background = 0, unknown = 128, foreground = 255
    
    void fitGMMs();

    // Run Grabcut refinement on the hard segmentation
    void refine() {int changed = 1; while (changed) changed = refineOnce();}
    int refineOnce();

    // Edit Trimap, mask values should be 0 or 255
    void setTrimap(cv::Mat &mask1u, const TrimapValue t) {_trimap1i.setTo(t, mask1u);}

    // Get Trimap for effective interaction. Format: CV_32SC1. Values should be TrimapValue
    cv::Mat& getTrimap() { return _trimap1i; }
    void getTrimap(cv::Mat &sal1f, cv::Mat& trimap, float t1, float t2);

    // Draw result
    void drawResult(cv::Mat& maskForeGround) 
    { 
        cv::compare(_segVal1f, 0.5, maskForeGround, CMP_GT);
    }

    cv::Mat showMedialResults(std::string& title);

private:

    int _w, _h;  // Width and height of the source image
    cv::Mat _imgBGR3f, _imgLab3f; // BGR images is used to find GMMs and Lab for pixel distance
    cv::Mat _trimap1i; // Trimap value
    cv::Mat _segVal1f; // Hard segmentation with type SegmentationValue

    // Variables used in formulas from the paper.
    float _lambda;  // lambda = 50. This value was suggested the GrabCut paper.
    float _beta;  // beta = 1 / ( 2 * average of the squared color distances between all pairs of neighboring pixels (8-neighborhood) )
    float _L;   // L = a large value to force a pixel to be foreground or background
    GraphF *_graph;

    // Storage for N-link weights, each pixel stores links to only four of its 8-neighborhood neighbors.
    // This avoids duplication of links, while still allowing for relatively easy lookup.
    // First 4 directions in DIRECTION8 are: right, rightBottom, bottom, leftBottom.
    Mat_<Vec4f> _NLinks; 

    int _directions[4]; // From DIRECTION8 for easy location

    CmGMM _bGMM, _fGMM; // Background and foreground GMM

    // Background and foreground GMM components, supply memory for GMM, 
    // not used for Grabcut
    cv::Mat _bGMMidx1i, _fGMMidx1i; 
    cv::Mat _show3u; // Image for display medial results
    
    // Update hard segmentation after running GraphCut, 
    // Returns the number of pixels that have changed from foreground to background or vice versa.
    int updateHardSegmentation();  

    void initGraph(); // builds the graph for GraphCut

    // Return number of difference and then expand fMask to get mask1u.
    static int ExpandMask(cv::Mat &fMask, cv::Mat &mask1u, cv::Mat &bdReg1u, int expandRatio = 5);
};

