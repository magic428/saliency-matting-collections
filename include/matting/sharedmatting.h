#ifndef SHAREDMSTTING_H
#define SHAREDMSTTING_H

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>
#include <vector>
//#include <cv.h>
//#include <highgui.h>

using namespace std;

struct labelPoint
{
    int x;
    int y;
    int label;
};

struct Tuple
{
    cv::Scalar f;
    cv::Scalar b;
    double   sigmaf;
    double   sigmab;

    int flag;

};

struct Ftuple
{
    cv::Scalar f;
    cv::Scalar b;
    double   alphar;
    double   confidence;
};

/*程序中认定cv::Point中 x为行，y为列，可能错误，但对程序结果没有影响*/
class SharedMatting
{
public:
    SharedMatting();
    ~SharedMatting();

    void loadImage(char * filename);
    void setImage(const cv::Mat &img);
    void loadTrimap(char * filename);
    void setTrimap(const cv::Mat &trimap);
    void expandKnown();
    void sample(cv::Point p, vector<cv::Point>& f, vector<cv::Point>& b);
    void gathering();
    void refineSample();
    void localSmooth();
    void solveAlpha();
    void save(char * filename);
    void Sample(vector<vector<cv::Point> > &F, vector<vector<cv::Point> > &B);
    void getMatte();
    cv::Mat& getMask() { return matte; };
    void release();

    double mP(int i, int j, cv::Scalar f, cv::Scalar b);
    double nP(int i, int j, cv::Scalar f, cv::Scalar b);
    double eP(int i1, int j1, int i2, int j2);
    double pfP(cv::Point p, vector<cv::Point>& f, vector<cv::Point>& b);
    double aP(int i, int j, double pf, cv::Scalar f, cv::Scalar b);
    double gP(cv::Point p, cv::Point fp, cv::Point bp, double pf);
    double gP(cv::Point p, cv::Point fp, cv::Point bp, double dpf, double pf);
    double dP(cv::Point s, cv::Point d);
    double sigma2(cv::Point p);
    double distanceColor2(cv::Scalar cs1, cv::Scalar cs2);
    double comalpha(cv::Scalar c, cv::Scalar f, cv::Scalar b);



private:
//    IplImage * pImg;
//    IplImage * trimap;
//    IplImage * matte;
    cv::Mat pImg;
    cv::Mat trimap;
    cv::Mat matte;

    vector<cv::Point> uT;         // uT Expand 后依然是 Unknown 区域的像素点
    vector<struct Tuple> tuples;  // Gathering 阶段得到的四元素元组 
    vector<struct Ftuple> ftuples;// Refinement 阶段得到的四元素元组 

    int height;
    int width;
    int kI;
    int kG;
    int ** unknownIndex; // Unknown 的索引信息；
    int ** tri;   // 保存 Expand 后重新标记过的 trimap, 在 loadImage() 中申请内存
    int ** alpha;
    double kC;    // Expand 阶段采用的邻域大小  

    int step;
    int channels;
    uchar* data;

};



#endif
