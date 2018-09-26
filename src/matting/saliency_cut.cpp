#include "matting/saliency_cut.h"
#include "saliency/saliency_rc.h"

CmSalCut::CmSalCut(cv::Mat &img3f)
    :_fGMM(5), _bGMM(5), _w(img3f.cols), _h(img3f.rows), _lambda(50)
{
    CV_Assert(img3f.data != NULL && img3f.type() == CV_32FC3);
    _imgBGR3f = img3f;
    cvtColor(_imgBGR3f, _imgLab3f, CV_BGR2Lab);
    _trimap1i = cv::Mat::zeros(_h, _w, CV_32S);
    _segVal1f = cv::Mat::zeros(_h, _w, CV_32F);
    _graph = NULL;
    
    _L = 8 * _lambda + 1;// Compute L
    _beta = 0; {// compute beta
        int edges = 0;
        double result = 0;
        for (int y = 0; y < _h; ++y) {
            const Vec3f* img = _imgLab3f.ptr<Vec3f>(y);
            for (int x = 0; x < _w; ++x){
                Point pnt(x, y);
                for (int i = 0; i < 4; i++)	{
                    Point pntN = pnt + DIRECTION8[i];
                    if (CHK_IND(pntN))
                        result += vecSqrDist(_imgLab3f.at<Vec3f>(pntN), img[x]), edges++;
                }
            }
        }
        _beta = (float)(0.5 * edges/result);
    }
    _NLinks.create(_h, _w); {// computeNLinks
        static const float dW[4] = {1, (float)(1/SQRT2), 1, (float)(1/SQRT2)};
        for (int y = 0; y < _h; y++) {
            Vec4f *nLink = _NLinks.ptr<Vec4f>(y);
            const Vec3f* img = _imgLab3f.ptr<Vec3f>(y);
            for (int x = 0; x < _w; x++, nLink++) {
                Point pnt(x, y);
                const Vec3f& c1 = img[x];
                for (int i = 0; i < 4; i++)	{
                    Point pntN = pnt + DIRECTION8[i];
                    if (CHK_IND(pntN))
                        (*nLink)[i] = _lambda * dW[i] * exp(-_beta * vecSqrDist(_imgLab3f.at<Vec3f>(pntN), c1));
                }
            }
        }
    }

    for (int i = 0; i < 4; i++)
        _directions[i] = DIRECTION8[i].x + DIRECTION8[i].y * _w;
}

CmSalCut::~CmSalCut(void)
{
    if (_graph)
        delete _graph;
}

cv::Mat CmSalCut::CutObjs(cv::Mat &_img3f, cv::Mat &_sal1f, cv::Mat &_border1u, 
    float t1, float t2, int wkSize)
{
    cv::Mat border1u = _border1u;
    if (border1u.data == NULL || border1u.size != _img3f.size){
        int bW = cvRound(0.02 * _img3f.cols), bH = cvRound(0.02 * _img3f.rows);
        border1u.create(_img3f.rows, _img3f.cols, CV_8U);
        border1u = 255;
        border1u(Rect(bW, bH, _img3f.cols - 2*bW, _img3f.rows - 2*bH)).setTo(0);
    }
    cv::Mat sal1f, wkMask; 
    _sal1f.copyTo(sal1f);
    sal1f.setTo(0.0, border1u);

    cv::Rect rect(0, 0, _img3f.cols, _img3f.rows);
    if (wkSize > 0){ 
        threshold(sal1f, sal1f, t1, 1, THRESH_TOZERO);
        sal1f.convertTo(wkMask, CV_8UC1, 255);
        threshold(wkMask, wkMask, 70, 255, THRESH_TOZERO);
        wkMask = CmCv::GetNZRegionsLS(wkMask, 0.005);
        if (wkMask.data == NULL)
            return cv::Mat();
        rect = CmCv::GetMaskRange(wkMask, wkSize);
        sal1f = sal1f(rect);
        border1u = border1u(rect);
        wkMask = wkMask(rect);
    }
    cv::Mat img3f = _img3f(rect); 
    
    cv::Mat fMask;
    CmSalCut salCut(img3f);
    salCut.initialize(sal1f, t1, t2);
    // cv::imshow("trimap", salCut.getTrimap()*255);
    // cv::waitKey();
    const int outerIter = 4;
    //salCut.showMedialResults("Ini");
    std::string title = std::string("Medial results");
    for (int j = 0; j < outerIter; j++)	{
        salCut.fitGMMs(); 
        int changed = 1000, times = 8;
        while (changed > 50 && times--) {
            // salCut.showMedialResults(title);
            changed = salCut.refineOnce();
            // waitKey();
        }
        title = format("It%d", j);
        // salCut.showMedialResults(title);
        // waitKey();
        salCut.drawResult(fMask);

        fMask = CmCv::GetNZRegionsLS(fMask);
        if (fMask.data == NULL)
            return cv::Mat();

        if (j == outerIter - 1 || ExpandMask(fMask, wkMask, border1u, 5) < 10)
            break;

        salCut.initialize(wkMask);
        fMask.copyTo(wkMask);
    }

    cv::Mat resMask = cv::Mat::zeros(_img3f.size(), CV_8U);
    fMask.copyTo(resMask(rect));
    return resMask;
}

/** 
 * \brief: Initialize using saliency map. 
 * 
 * In the Trimap: background < t1, foreground > t2, others unknown.
 * 
 * Saliency values are in [0, 1], "sal1f" and "1-sal1f" are used as weight 
 * to train fore and back ground GMMs
 */
void CmSalCut::initialize(cv::Mat &sal1f, float t1, float t2)
{
    CV_Assert(sal1f.type() == CV_32FC1 && sal1f.size == _imgBGR3f.size);
    sal1f.copyTo(_segVal1f);

    for (int y = 0; y < _h; y++) {
        int* triVal = _trimap1i.ptr<int>(y);
        const float *segVal = _segVal1f.ptr<float>(y);
        for (int x = 0; x < _w; x++) {
            triVal[x] = segVal[x] < t1 ? TrimapBackground : TrimapUnknown;
            triVal[x] = segVal[x] > t2 ? TrimapForeground : triVal[x]; 
        }
    }
}

void CmSalCut::getTrimap(cv::Mat &sal1f, cv::Mat& trimap, float t1, float t2)
{
    // std::cout << "sal1f must be CV_32FC1... " << std::endl;
    CV_Assert(sal1f.type() == CV_32FC1 && sal1f.size == _imgBGR3f.size);
    sal1f.copyTo(_segVal1f);
    if(!trimap.data)
        trimap = cv::Mat::zeros(sal1f.rows, sal1f.cols, CV_8UC1);
    size_t n_unknown_pixels = 0;

    for (int y = 0; y < _h; y++) {
        uchar* triVal = trimap.ptr<uchar>(y);
        const float *segVal = _segVal1f.ptr<float>(y);
        for (int x = 0; x < _w; x++) {
            triVal[x] = segVal[x] < t1 ? TrimapBackground : TrimapUnknown;
            triVal[x] = segVal[x] > t2 ? TrimapForeground : triVal[x]; 
            if ( TrimapUnknown == triVal[x] )
                ++n_unknown_pixels;
        }
    }

    std::cout << "unknown pixels: " << n_unknown_pixels << std::endl;
    std::cout << "unknown pixels percent: " << (float)n_unknown_pixels/(sal1f.cols*sal1f.rows)*100 << "%" << std::endl;

    if ( n_unknown_pixels > size_t(0.05*sal1f.cols*sal1f.rows) ) 
        getTrimap( sal1f, trimap, t1+0.05, t2-0.05);
}

void CmSalCut::initialize(cv::Mat &sal1u) // Background = 0, unknown = 128, foreground = 255
{
    std::cout << "sal1u must be CV_8UC1... " << std::endl;
    CV_Assert(sal1u.type() == CV_8UC1 && sal1u.size == _imgBGR3f.size);
    for (int y = 0; y < _h; y++) {
        int* triVal = _trimap1i.ptr<int>(y);
        const byte *salVal = sal1u.ptr<byte>(y);
        float *segVal = _segVal1f.ptr<float>(y);
        for (int x = 0; x < _w; x++) {
            triVal[x] = salVal[x] < 70 ? TrimapBackground : TrimapUnknown;
            triVal[x] = salVal[x] > 200 ? TrimapForeground : triVal[x]; 
            segVal[x] = salVal[x] < 70 ? 0 : 1.f;
        }
    }
}

// Initial rect region in between thr1 and thr2 and others below thr1 as the Grabcut paper 
void CmSalCut::initialize(const Rect &rect)
{
    _trimap1i = TrimapBackground;
    _trimap1i(rect) = TrimapUnknown;
    _segVal1f.setTo(0);
    _segVal1f(rect) = 1;
}

void CmSalCut::fitGMMs()
{
    _fGMM.BuildGMMs(_imgBGR3f, _fGMMidx1i, _segVal1f);

    cv::Mat dst, ones = cv::Mat::ones(_segVal1f.size(), _segVal1f.type());
    cv::subtract(ones, _segVal1f, dst);
    _bGMM.BuildGMMs(_imgBGR3f, _bGMMidx1i, dst);
}

int CmSalCut::refineOnce()
{
    // Steps 4 and 5: Learn new GMMs from current segmentation
    if (_fGMM.GetSumWeight() < 50 || _bGMM.GetSumWeight() < 50)
        return 0;

    _fGMM.RefineGMMs(_imgBGR3f, _fGMMidx1i, _segVal1f);
    cv::Mat dst, ones = cv::Mat::ones(_segVal1f.size(), _segVal1f.type());
    cv::subtract(ones, _segVal1f, dst);
    _bGMM.BuildGMMs(_imgBGR3f, _bGMMidx1i, dst);

    // Step 6: Run GraphCut and update segmentation
    initGraph();
    if (_graph)
        _graph->maxflow();

    return updateHardSegmentation();
}

int CmSalCut::updateHardSegmentation()
{
    int changed = 0;
    for (int y = 0, id = 0; y < _h; ++y) {
        float* segVal = _segVal1f.ptr<float>(y);
        int* triMapD = _trimap1i.ptr<int>(y);
        for (int x = 0; x < _w; ++x, id++) {
            float oldValue = segVal[x];
            if (triMapD[x] == TrimapBackground)
                segVal[x] = 0.f; // SegmentationBackground
            else if (triMapD[x] == TrimapForeground)
                segVal[x] = 1.f; // SegmentationForeground
            else 
                segVal[x] = _graph->what_segment(id) == GraphF::SOURCE ? 1.f : 0.f;
            changed += abs(segVal[x] - oldValue) > 0.1 ? 1 : 0;
        }
    }
    return changed;
}

void CmSalCut::initGraph()
{
    // Set up the graph (it can only be used once, so we have to recreate it each time the graph is updated)
    if (_graph == NULL)
        _graph = new GraphF(_w * _h, 4 * _w * _h);
    else
        _graph->reset();
    _graph->add_node(_w * _h);

    for (int y = 0, id = 0; y < _h; ++y) {
        int* triMapD = _trimap1i.ptr<int>(y);
        const float* img = _imgBGR3f.ptr<float>(y);
        for(int x = 0; x < _w; x++, img += 3, id++) {
            float back, fore;
            if (triMapD[x] == TrimapUnknown ) {
                fore = -log(_bGMM.P(img));
                back = -log(_fGMM.P(img));
            }
            else if (triMapD[x] == TrimapBackground ) 
                fore = 0, back = _L;
            else		// TrimapForeground
                fore = _L,	back = 0;
            
            // Set T-Link weights
            _graph->add_tweights(id, fore, back); // _graph->set_tweights(_nodes(y, x), fore, back);

            // Set N-Link weights from precomputed values
            Point pnt(x, y);
            const Vec4f& nLink = _NLinks(pnt);
            for (int i = 0; i < 4; i++)	{
                Point nPnt = pnt + DIRECTION8[i];
                if (CHK_IND(nPnt))
                    _graph->add_edge(id, id + _directions[i], nLink[i], nLink[i]);
            }
        }
    }
}

cv::Mat CmSalCut::showMedialResults(std::string& title)
{
    _show3u.create(_h, _w * 2, CV_8UC3);
    cv::Mat showTri = _show3u(Rect(0, 0, _w, _h));
    cv::Mat showSeg = _show3u(Rect(_w, 0, _w, _h));
    _imgBGR3f.convertTo(showTri, CV_8U, 255);
    showTri.copyTo(showSeg);

    for (int y = 0; y < _h; y++){
        const int* triVal = _trimap1i.ptr<int>(y);
        const float* segVal = _segVal1f.ptr<float>(y);
        Vec3b* triD = showTri.ptr<Vec3b>(y);
        Vec3b* segD = showSeg.ptr<Vec3b>(y);
        for (int x = 0; x < _w; x++, triD++, segD++) {
            switch (triVal[x]){
            case TrimapForeground: (*triD)[2] = 255; break; // Red
            case TrimapBackground: (*triD)[1] = 255; break; // Green
            }
            if (segVal[x] > 0.5)
                (*segD)[0] = 255;
            else
                (*segD) /= 2;
        }
    }
    cv::imshow(title, _show3u);
    return _show3u;
}

int CmSalCut::ExpandMask(cv::Mat &fMask, cv::Mat &mask1u, cv::Mat &bdReg1u, int expandRatio)
{
    compare(fMask, mask1u, mask1u, CMP_NE);
    int changed = cvRound(sum(mask1u).val[0] / 255.0);

    cv::Mat bigM, smalM;
    dilate(fMask, bigM, cv::Mat(), Point(-1, -1), expandRatio);
    erode(fMask, smalM, cv::Mat(), Point(-1, -1), expandRatio);
    static const double erodeSmall = 255 * 50;
    if (sum(smalM).val[0] < erodeSmall)
        smalM = fMask;
    mask1u = bigM * 0.5 + smalM * 0.5;
    mask1u.setTo(0, bdReg1u);
    return changed;
}

/**
 * \brief: Salient Object Detection and segment foregrond
 * 
 * \param: 
 * 
*/
int CmSalCut::Demo(std::string root_dir, std::string salDir)
{

    std::vector<std::string> names; 
    std::vector<std::string> paths; 
    std::string inDir, ext;

    // Get files' path and number in the "root_dir" dirctory
    int imgNum = FileOps::get_image_paths(root_dir, paths, false);
    // Get files' name without extension
    FileOps::get_image_names_without_ext(root_dir, names, false);

    // Create the output directory
    if(salDir.back() != '/')
        salDir += "/";
    FileOps::mkdir(salDir);
    std::cout << "Get saliency maps for images " << root_dir.c_str() <<
                 "\n And save results to " << salDir.c_str() << std::endl;

    // CmTimer tm("Saliency detection and segmentation");
    // tm.Start();

#pragma omp parallel for 
    /**
     * Salient Object Detection based on Global contrast
     *
     *
    */
    for (int i = 0; i < imgNum; i++){

        string image_path = paths[i];
        string saliency_map_path = salDir + names[i] + "_RCC.png";
        string origin_image_path = salDir + names[i] + ".png";
        cv::Mat image, sal_gmr;

        if (FileOps::exists(saliency_map_path))
            continue;

        std::cout << "Processing " << i << "/" << imgNum << "th image: " << image_path << std::endl;

        cv::Mat img3f = cv::imread(image_path);
        CV_Assert_(img3f.data != NULL, ("Can't load image %s\n", image_path.c_str()));
        image = img3f.clone();
        img3f.convertTo(img3f, CV_32FC3, 1.0/255);
        
        // get salient map
        cv::Mat sal = Mat::zeros(img3f.rows, img3f.cols, CV_8UC1);
        // get_saliency_map(img3f, sal);
        sal = CmSaliencyRC::GetRC(img3f);
        sal.convertTo(sal, CV_32FC1, 1.0/255);

        /**
         *  GMR: Get Saliency Map
         */
        // GMRsaliency GMRsal;
        // sal_gmr = GMRsal.GetSal(image);
        // sal_gmr.convertTo(sal, CV_32FC1, 1.0/255);

        // get saliency mask 
        cv::Mat cutMask;
        cv::Mat foreground(image.size(), CV_8UC3, cv::Scalar(255, 255, 255));
        float t = 0.8f;  // recall rate
        int maxIt = 4;
        cv::GaussianBlur(sal, sal, Size(9, 9), 0);  
        cv::normalize(sal, sal, 0, 1, NORM_MINMAX);
        cv::Mat borderMask = cv::Mat(); 
        while (cutMask.empty() && maxIt--){
            cutMask = CmSalCut::CutObjs(img3f, sal, borderMask, 0.1f, t);
            t -= 0.2f;
        }

        if (!cutMask.empty()){
            image.copyTo(foreground, cutMask);
            // cv::imwrite(origin_image_path, image);
            // cv::imwrite(saliency_map_path, foreground);
            cv::imshow("input", image);
            cv::imshow("foreground", foreground);
            cv::imshow("mask", cutMask);

            if (cv::waitKey() == 27){

                cv::destroyAllWindows(); 
                break;
            }
        }
    }

    // tm.Stop();
    std::cout << "Salient object detection and segmentation finished" << std::endl;
            //   << tm.TimeInSeconds()/imgNum << " seconds used per image" << std::endl;
    return 0;
}
