#include "matting/sharedmatting.h"
#include <time.h>

using namespace cv;

//构造函数
SharedMatting::SharedMatting()
{
    kI = 10;
    kC = 5.0;
    kG = 4;  //each unknown p gathers at most kG forground and background samples
    uT.clear();
    tuples.clear();

}

// 析构函数
SharedMatting::~SharedMatting()
{
//    cvReleaseImage(&pImg);
//    cvReleaseImage(&trimap);
//    cvReleaseImage(&matte);
    pImg.release();
    trimap.release();
    matte.release();
    uT.clear();
    tuples.clear();
    ftuples.clear();

    for (int i = 0; i < height; ++i)
    {
        delete[] tri[i];
        delete[] unknownIndex[i];
        delete[] alpha[i];
    }
    delete[] tri;
    delete[] unknownIndex;
    delete[] alpha;
}


//载入图像
void SharedMatting::loadImage(char * filename)
{
    pImg = imread(filename);
    if (!pImg.data)
    {
        cout << "Loading Image Failed!" << endl;
        exit(-1);
    }

    //height     = pImg->height;
    height     = pImg.rows;
    //width      = pImg->width;
    width      = pImg.cols;
    //step       = pImg->widthStep;
    step       = pImg.step1();
//    channels   = pImg->nChannels;
    channels   = pImg.channels();
//    data    = (uchar *)pImg->imageData;
    data       = (uchar *)pImg.data;

    unknownIndex  = new int*[height];
    tri           = new int*[height];
    alpha         = new int*[height];
    for(int i = 0; i < height; ++i)
    {
        unknownIndex[i] = new int[width];
        tri[i]          = new int[width];
        alpha[i]        = new int[width];
    }

//    matte = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
    matte.create(Size(width, height), CV_8UC1);
}

void SharedMatting::setImage(const cv::Mat& img)
{
    // std::cout << "img type must be CV_8UC3... " << std::endl;
    CV_Assert(img.type() == CV_8UC3);

    if (!img.data) {
        cout << "Invalid Image Mat!" << endl;
        exit(-1);
    }

    img.copyTo(pImg);

    height     = pImg.rows;
    width      = pImg.cols;
    step       = pImg.step1();
    channels   = pImg.channels();
    data       = (uchar *)pImg.data;
    
    unknownIndex  = new int*[height];
    tri           = new int*[height];
    alpha         = new int*[height];
    for(int i = 0; i < height; ++i)
    {
        unknownIndex[i] = new int[width];
        tri[i]          = new int[width];
        alpha[i]        = new int[width];
    }

    matte.create(Size(width, height), CV_8UC1);
}


// 载入第三方图像
void SharedMatting::loadTrimap(char * filename)
{
    trimap = imread(filename);
    if (!trimap.data) {
        cout << "Loading Trimap Failed!" << endl;
        exit(-1);
    }

    /*cvNamedWindow("aa");
    cvShowImage("aa", trimap);
    cvWaitKey(0);*/
}

void SharedMatting::setTrimap(const cv::Mat& triMap)
{
    // std::cout << "triMap must be CV_8UC3... " << std::endl;
    CV_Assert(triMap.type() == CV_8UC3);

    if (!triMap.data) {
        cout << "Invalid Trimap!" << endl;
        exit(-1);
    }

    triMap.copyTo(trimap);
    // cv::resize(trimap, trimap, cv::Size(triMap.cols/2, triMap.rows/2));

    /*cvNamedWindow("aa");
    cvShowImage("aa", trimap);
    cvWaitKey(0);*/
}

/**
 * \brief: 扩展图像的未知区域, 属于预处理, 为了减少后期的计算量
 * 
 *  作用原理: 在一定的邻域内 (kI = 10) 计算色彩相似度, 如果色彩相似度满足一定条件, 
 *           那就将其归属到相似的像素区域: 背景或者前景.  
 * 
 * \Trick: 色彩相似度的计算采用欧氏距离, 为了提高算法效率, 并未对距离取平方, 
 *         因此在比较是否满足条件时使用的是 kc2 = kC * kC; 
*/
void SharedMatting::expandKnown()
{
    int kc2 = kC * kC;
    int s       = trimap.step1();
    int c       = trimap.channels();
    uchar * d   = (uchar *)trimap.data;
    vector<struct labelPoint> vp;
    vp.clear();

    // 缩小 trimap 中的 Unknown 区域, 即重新标记该区域内的像素值
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            tri[i][j] = d[i * step + j * channels];
        }
    }
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {

            if (tri[i][j] != 0 && tri[i][j] != 255) {

                int label = -1;
                double dmin = 10000.0;
                bool flag = false;      // 标记该像素已经被分配到背景/前景 
                int pb = data[i * step + j * channels];
                int pg = data[i * step + j * channels + 1];
                int pr = data[i * step + j * channels + 2];
                Scalar p = Scalar(pb, pg, pr);

                // 在半径为 10 的邻域内计算色彩相似度, 从正方形的中心开始向外层扩展
                for (int k = 0; (k <= kI) && !flag; ++k) {

                    // 构建正方形邻域
                    int k1 = max(0, i - k);
                    int k2 = min(i + k, height - 1);
                    int l1 = max(0, j - k);
                    int l2 = min(j + k, width - 1);

                    // 先按行外扩搜索
                    for (int l = k1; (l <= k2) && !flag; ++l) {

                        double dis;
                        double gray = tri[l][l1];

                        // 只和背景/前景区域内的像素比较
                        if (gray == 0 || gray == 255) {

                            // for 循环已经保证了在合适距离范围内, 这一步是否多余? 
                            dis = dP(Point(i, j), Point(l, l1));  
                            if (dis > kI)
                                continue;

                            int qb = data[l * step + l1 * channels];
                            int qg = data[l * step + l1 * channels + 1];
                            int qr = data[l * step + l1 * channels + 2];
                            Scalar q = Scalar(qb, qg, qr);

                            double distanceColor = distanceColor2(p, q);
                            if (distanceColor <= kc2) {
                                flag = true;
                                label = gray;
                            }
                        }
                        if (flag)
                            break;

                        gray = tri[l][l2];
                        if (gray == 0 || gray == 255) {

                            dis = dP(Point(i, j), Point(l, l2));
                            if (dis > kI)
                                continue;
                            int qb = data[l * step + l2 * channels];
                            int qg = data[l * step + l2 * channels + 1];
                            int qr = data[l * step + l2 * channels + 2];
                            Scalar q = Scalar(qb, qg, qr);

                            double distanceColor = distanceColor2(p, q);
                            if (distanceColor <= kc2) {
                                flag = true;
                                label = gray;
                            }
                        }
                    }

                    // 再按列外扩搜索
                    for (int l = l1; (l <= l2) && !flag; ++l) {
                        double dis;
                        double gray;

                        gray = tri[k1][l];
                        if (gray == 0 || gray == 255) {

                            dis = dP(Point(i, j), Point(k1, l));
                            if (dis > kI)
                                continue;

                            int qb = data[k1 * step + l * channels];
                            int qg = data[k1 * step + l * channels + 1];
                            int qr = data[k1 * step + l * channels + 2];
                            Scalar q = Scalar(qb, qg, qr);

                            double distanceColor = distanceColor2(p, q);
                            if (distanceColor <= kc2) {
                                flag = true;
                                label = gray;
                            }
                        }

                        gray = tri[k2][l];
                        if (gray == 0 || gray == 255) {

                            dis = dP(Point(i, j), Point(k2, l));
                            if (dis > kI)
                                continue;

                            int qb = data[k2 * step + l * channels];
                            int qg = data[k2 * step + l * channels + 1];
                            int qr = data[k2 * step + l * channels + 2];
                            Scalar q = Scalar(qb, qg, qr);

                            double distanceColor = distanceColor2(p, q);
                            if (distanceColor <= kc2) {
                                flag = true;
                                label = gray;
                            }
                        }
                    }
                }

                // 检查是否被重新标记
                // 使用 vp 中转的原因:扩展只是预处理, 不能及时更新, 不然会导致大面积
                // 未知区域像素被重新标记为前景/背景像素的情况发生 
                if (label != -1) {
                
                    struct labelPoint lp;
                    lp.x = i;
                    lp.y = j;
                    lp.label = label;
                    vp.push_back(lp);
                } else {

                    Point lp;
                    lp.x = i;
                    lp.y = j;
                    uT.push_back(lp);
                }
            }
        }
    }

    vector<struct labelPoint>::iterator it;
    for (it = vp.begin(); it != vp.end(); ++it) {
        int ti = it->x;
        int tj = it->y;
        int label = it->label;
        //cvSet2D(trimap, ti, tj, ScalarAll(label));
        tri[ti][tj] = label;
    }
    vp.clear();
    // cvNamedWindow("trimap");
    // cvShowImage("trimap", trimap);
    // cvWaitKey(0);
}

/**
 * \brief: Compute alpha value
 * 
*/
double SharedMatting::comalpha(Scalar c, Scalar f, Scalar b)
{
    double alpha = ((c.val[0] - b.val[0]) * (f.val[0] - b.val[0]) +
                    (c.val[1] - b.val[1]) * (f.val[1] - b.val[1]) +
                    (c.val[2] - b.val[2]) * (f.val[2] - b.val[2]))
                 / ((f.val[0] - b.val[0]) * (f.val[0] - b.val[0]) +
                    (f.val[1] - b.val[1]) * (f.val[1] - b.val[1]) +
                    (f.val[2] - b.val[2]) * (f.val[2] - b.val[2]) + 0.0000001);

    return min(1.0, max(0.0, alpha));
}

double SharedMatting::mP(int i, int j, Scalar f, Scalar b)
{
    int bc = data[i * step + j * channels];
    int gc = data[i * step + j * channels + 1];
    int rc = data[i * step + j * channels + 2];
    Scalar c = Scalar(bc, gc, rc);

    double alpha = comalpha(c, f, b);

    double result = sqrt((c.val[0] - alpha * f.val[0] - (1 - alpha) * b.val[0]) * (c.val[0] - alpha * f.val[0] - (1 - alpha) * b.val[0]) +
                         (c.val[1] - alpha * f.val[1] - (1 - alpha) * b.val[1]) * (c.val[1] - alpha * f.val[1] - (1 - alpha) * b.val[1]) +
                         (c.val[2] - alpha * f.val[2] - (1 - alpha) * b.val[2]) * (c.val[2] - alpha * f.val[2] - (1 - alpha) * b.val[2]));
    return result / 255.0;
}

double SharedMatting::nP(int i, int j, Scalar f, Scalar b)
{
    int i1 = max(0, i - 1);
    int i2 = min(i + 1, height - 1);
    int j1 = max(0, j - 1);
    int j2 = min(j + 1, width - 1);

    double  result = 0;

    for (int k = i1; k <= i2; ++k)
    {
        for (int l = j1; l <= j2; ++l)
        {
            double m = mP(k, l, f, b);
            result += m * m;
        }
    }

    return result;
}

/**
 * \brief: Ep(s): 计算从 p 到达采样点所需的能量, 公式(4)
 * 
 * \param: px, py - 点 p 的坐标; 
 *         sx, sy - 采样点的坐标
 * 
 * \return: 返回所需的能量值
*/
double SharedMatting::eP(int px, int py, int sx, int sy)
{
    // 计算向量 (s - p)
    double cx = sx - px;  
    double cy = sy - py;
    double norm  = sqrt(cx * cx + cy * cy);

    // 归一化向量 (s - p)
    double cx_norm = cx / (norm + 0.0000001);
    double cy_norm = cy / (norm + 0.0000001);

    // 积分求解: 使用无限分割法累加
    // 分割的间距: step = 1 / (abs(cx_norm) 或 1 / (abs(cy_norm)
    double step = min(1 / (abs(cx_norm) + 1e-10), 1 / (abs(cy_norm) + 1e-10));
    int b = data[int(px * step) + py * channels];
    int g = data[int(px * step) + py * channels + 1];
    int r = data[int(px * step) + py * channels + 2];
    Scalar pre = Scalar(b, g, r);
    double result = 0;
    int ti = px;
    int tj = py;
    for ( int k = 1; ; ++k ) {

        double delta_x = cx_norm * (1 + step * k);
        double delta_y = cy_norm * (1 + step * k);
        int i = int(px + delta_x + 0.5);
        int j = int(py + delta_y + 0.5);

        double dr = 1;

        int b = data[int(i * step) + j * channels];
        int g = data[int(i * step) + j * channels + 1];
        int r = data[int(i * step) + j * channels + 2];
        Scalar cur = Scalar(b, g, r);

        // 近似计算 dr 的代码, 因为是无限分割, 因此梯度的投影可以近似等于向量
        if (ti - i > 0 && tj - j == 0)
            dr = cy_norm;
        else if(ti - i == 0 && tj - j > 0)
            dr = cx_norm;

        // 梯度的计算方法: 各个通道像素差. 
        // dr 表示归一化的方向向量?
        // result += ((cur.val[0] - pre.val[0]) * (cur.val[0] - pre.val[0]) +
        //            (cur.val[1] - pre.val[1]) * (cur.val[1] - pre.val[1]) +
        //            (cur.val[2] - pre.val[2]) * (cur.val[2] - pre.val[2])) * dr;
        int namla_I = ((cur.val[0] - pre.val[0]) * (cur.val[0] - pre.val[0]) +
                   (cur.val[1] - pre.val[1]) * (cur.val[1] - pre.val[1]) +
                   (cur.val[2] - pre.val[2]) * (cur.val[2] - pre.val[2]));
        result += (namla_I * cx_norm * cx_norm) + (namla_I * cy_norm * cy_norm);

        pre = cur;
        ti = i;
        tj = j;

        // 原始代码中这个结束条件是这样的: 
        // if(abs(cx) >= abs(delta_x) || abs(cy) >= abs(delta_y)) break;
        if(abs(delta_x) >= abs(cx) || abs(delta_y) >= abs(cy))
            break;
    }

    return result;
}


/**
 * \brief: PF_p: 计算 p 属于前景的概率, 公式(5)
 * 
 * \param: p - 点 p 的坐标; 
 *         f - 采样得到的前景点集合
 *         b - 采样得到的背景点集合
 * 
 * \return: 返回 p 属于前景的概率
*/
double SharedMatting::pfP(Point p, vector<Point>& f, vector<Point>& b)
{
    double fmin = 1e10;
    vector<Point>::iterator it;

    for (it = f.begin(); it != f.end(); ++it) {

        double fp = eP(p.x, p.y, it->x, it->y);
        if (fp < fmin)
            fmin = fp;
    }

    double bmin = 1e10;
    for (it = b.begin(); it != b.end(); ++it) {

        double bp = eP(p.x, p.y, it->x, it->y);
        if (bp < bmin)
            bmin = bp;
    }

    return bmin / (fmin + bmin + 1e-10);
}

double SharedMatting::aP(int i, int j, double pf, Scalar f, Scalar b)
{
    int bc = data[i * step + j * channels];
    int gc = data[i * step + j * channels + 1];
    int rc = data[i * step + j * channels + 2];
    Scalar c = Scalar(bc, gc, rc);

    double alpha = comalpha(c, f, b);

    return pf + (1 - 2 * pf) * alpha;
}

double SharedMatting::dP(Point s, Point d)
{
    return sqrt(double((s.x - d.x) * (s.x - d.x) + (s.y - d.y) * (s.y - d.y)));
}

double SharedMatting::gP(Point p, Point fp, Point bp, double pf)
{
    int bc, gc, rc;
    bc = data[fp.x * step + fp.y * channels];
    gc = data[fp.x * step + fp.y * channels + 1];
    rc = data[fp.x * step + fp.y * channels + 2];
    Scalar f = Scalar(bc, gc, rc);
    bc = data[bp.x * step + bp.y * channels];
    gc = data[bp.x * step + bp.y * channels + 1];
    rc = data[bp.x * step + bp.y * channels + 2];
    Scalar b = Scalar(bc, gc, rc);


    double tn = pow(nP(p.x, p.y, f, b), 3);
    double ta = pow(aP(p.x, p.y, pf, f, b), 2);
    double tf = dP(p, fp);
    double tb = pow(dP(p, bp), 4);

    //cout << "tn:" << tn << "ta:" << ta << "tf:" << tf << "tb:" << tb << endl;
    return tn * ta * tf * tb;

}

double SharedMatting::gP(Point p, Point fp, Point bp, double dpf, double pf)
{
    int bc, gc, rc;
    bc = data[fp.x * step + fp.y * channels];
    gc = data[fp.x * step + fp.y * channels + 1];
    rc = data[fp.x * step + fp.y * channels + 2];
    Scalar f = Scalar(bc, gc, rc);
    bc = data[bp.x * step + bp.y * channels];
    gc = data[bp.x * step + bp.y * channels + 1];
    rc = data[bp.x * step + bp.y * channels + 2];
    Scalar b = Scalar(bc, gc, rc);


    double tn = pow(nP(p.x, p.y, f, b), 3);
    double ta = pow(aP(p.x, p.y, pf, f, b), 2);
    double tf = dpf;
    double tb = pow(dP(p, bp), 4);

    return tn * ta * tf * tb;
}

double SharedMatting::sigma2(Point p)
{
    int xi = p.x;
    int yj = p.y;
    int bc, gc, rc;
    bc = data[xi * step + yj * channels];
    gc = data[xi * step + yj * channels + 1];
    rc = data[xi * step + yj * channels + 2];
    Scalar pc = Scalar(bc, gc, rc);

    int i1 = max(0, xi - 2);
    int i2 = min(xi + 2, height - 1);
    int j1 = max(0, yj - 2);
    int j2 = min(yj + 2, width - 1);

    double result = 0;
    int    num    = 0;

    for (int i = i1; i <= i2; ++i)
    {
        for (int j = j1; j <= j2; ++j)
        {
            int bc, gc, rc;
            bc = data[i * step + j * channels];
            gc = data[i * step + j * channels + 1];
            rc = data[i * step + j * channels + 2];
            Scalar temp = Scalar(bc, gc, rc);
            result += distanceColor2(pc, temp);
            ++num;
        }
    }

    return result / (num + 1e-10);

}

/**
 * \brief: 计算两个像素点色彩的欧式巨鹿
*/
double SharedMatting::distanceColor2(Scalar cs1, Scalar cs2)
{
    return (cs1.val[0] - cs2.val[0]) * (cs1.val[0] - cs2.val[0]) +
           (cs1.val[1] - cs2.val[1]) * (cs1.val[1] - cs2.val[1]) +
           (cs1.val[2] - cs2.val[2]) * (cs1.val[2] - cs2.val[2]);
}

void SharedMatting::sample(Point p, std::vector<Point> &f, std::vector<Point> &b)
{
    int i = p.x;
    int j = p.y;

    double inc   = 360.0 / kG;
    //cout << inc << endl;
    double ca    = inc / 9;
    double angle = (i % 3 * 3 + j % 9) * ca;
    for (int k = 0; k  < kG; ++k)
    {
        bool flagf = false;
        bool flagb = false;

        double z  = (angle + k * inc) / 180 * 3.1415926;
        double ei = sin(z);
        double ej = cos(z);

        double step = min(1.0 / (abs(ei) + 1e-10), 1.0 / (abs(ej) + 1e-10));

        for (double t = 1; ;t += step)
        {
            int ti = int(i + ei * t + 0.5);
            int tj = int(j + ej * t + 0.5);


            if(ti >= height || ti < 0 || tj >= width || tj < 0)
            {
                break;
            }
            int gray = tri[ti][tj];

            if (!flagf && gray == 255)
            {
                Point tp = Point(ti, tj);
                f.push_back(tp);
                flagf = true;
            }
            else if (!flagb && gray == 0)
            {
                Point tp = Point(ti, tj);
                b.push_back(tp);
                flagb = true;
            }
            if (flagf && flagb)
            {
                break;
            }

        }

    }

}

/**
 * \brief: 采样阶段, 在 kG (=4) 条路径上采样最近的背景点和前景点, 
 *         每条路径方向上只选择一个背景点和一个前景点, 共两个点.
 *         因此, 一共有 kG * kG 中前景点和背景点的组合.   
 * 
 * \param: tri - trimap 图
 * 
 * \note: (1) 采样得到的前景点和背景点的个数不一定相同
 *        (2) 采样得到的前景点个数或背景点的个数可能为 0 
 * 
 * \return: F - 采样得到的前景点
 *          B - 采样得到的背景点
*/
void SharedMatting::Sample(std::vector<vector<Point> > &F, std::vector<vector<Point> > &B)
{
    int   a,b,i;
    int   x,y,p,q;
    int   w,h,gray;
    int   angle;
    double z,ex,ey,t,step;

    a = 360 / kG;  // kG 条路径之间的夹角步长  
    b = 1.7f * a / 9;

    F.clear();
    B.clear();
    w= pImg.cols;
    h= pImg.rows;
    for( vector<Point>::iterator iter= uT.begin(); iter!= uT.end(); ++iter ) {

        vector<Point> fPts, bPts;

        x= iter->x;
        y= iter->y;
        angle= (x + y) * b % a;
        
        // 沿着 kG 条路径搜索 
        for(i = 0; i < kG; ++i) {
            
            bool got_bg(false), got_fg(false);

            z = (angle + i * a) / 180.0f * 3.1415926f;
            ex = sin(z);
            ey = cos(z);
            step = min(1.0f / (abs(ex)+1e-10f), 1.0f / (abs(ey)+1e-10f));

            for( t = 0; ; t += step ) {

                p = (int)(x + ex * t + 0.5f);
                q = (int)(y + ey * t + 0.5f);
                if(p < 0 || p >= h || q < 0 || q >= w)  // 到达边界  
                    break;

                gray = tri[p][q];
                if(!got_bg && gray < 50) {  // backgorund 

                    Point pt = Point(p, q);
                    bPts.push_back(pt);
                    got_bg=true;
                } else if(!got_fg && gray > 200) { // foregorund
                    
                        Point pt = Point(p, q);
                        fPts.push_back(pt);
                        got_fg=true;
                } else {   // unknown area

                        if(got_bg && got_fg)
                            break;
                }
            }
        }

        F.push_back(fPts);
        B.push_back(bPts);
    }
}

/**
 * \brief: Sample Gathering 阶段  
 * 
 * \param: F, B - 采样得到的前景点和背景点  
 * 
 * 在采样中根据目标函数选取最优采样点, 用来估计 alpah 的值  
 * 
 * \return: 
 * 
*/
void SharedMatting::gathering()
{
    vector<vector<Point> > F,B;
    Sample(F, B);

    int index = 0;
    double a;
    vector<Point> f;
    vector<Point> b;
    vector<Point>::iterator it;
    vector<Point>::iterator it1;
    vector<Point>::iterator it2;

    // 只处理 Unknown 区域的像素点
    for (int m = 0; m < uT.size(); ++m) {

        int i = uT[m].x;
        int j = uT[m].y;
        double pfp = pfP(Point(i, j), F[m], B[m]);
        double gmin = 1.0e10;

        Point best_f;
        Point best_b;

        bool flag = false;   // 找到最优的前景和背景采样点
        bool first = true;
        for (it1 = F[m].begin(); it1 != F[m].end(); ++it1) {
            
            double dpf = dP(Point(i, j), *(it1));
            for (it2 = B[m].begin(); it2 < B[m].end(); ++it2) {

                double gp = gP(Point(i, j), *(it1), *(it2), dpf, pfp);

                if (gp < gmin) {
                    gmin = gp;
                    best_f   = *(it1);
                    best_b   = *(it2);
                    flag = true;
                }
            }
        }

        struct Tuple st;
        st.flag = -1;
        if (flag) {
            int bc, gc, rc;
            bc = data[best_f.x * step +  best_f.y * channels];
            gc = data[best_f.x * step +  best_f.y * channels + 1];
            rc = data[best_f.x * step +  best_f.y * channels + 2];
            st.flag   = 1;
            st.f      = Scalar(bc, gc, rc);
            bc = data[best_b.x * step +  best_b.y * channels];
            gc = data[best_b.x * step +  best_b.y * channels + 1];
            rc = data[best_b.x * step +  best_b.y * channels + 2];
            st.b      = Scalar(bc, gc, rc);
            st.sigmaf = sigma2(best_f);
            st.sigmab = sigma2(best_b);
        }

        tuples.push_back(st);
        unknownIndex[i][j] = index;
        ++index;
    }
    f.clear();
    b.clear();

}

void SharedMatting::refineSample()
{
    ftuples.resize(width * height + 1);
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            int b, g, r;
            b = data[i * step +  j* channels];
            g = data[i * step +  j * channels + 1];
            r = data[i * step +  j * channels + 2];
            Scalar c = Scalar(b, g, r);
            int indexf = i * width + j;
            int gray = tri[i][j];
            if (gray == 0 )
            {
                ftuples[indexf].f = c;
                ftuples[indexf].b = c;
                ftuples[indexf].alphar = 0;
                ftuples[indexf].confidence = 1;
                alpha[i][j] = 0;
            }
            else if (gray == 255)
            {
                ftuples[indexf].f = c;
                ftuples[indexf].b = c;
                ftuples[indexf].alphar = 1;
                ftuples[indexf].confidence = 1;
                alpha[i][j] = 255;
            }

        }
    }
    vector<Point>::iterator it;
    for (it = uT.begin(); it != uT.end(); ++it)
    {
        int xi = it->x;
        int yj = it->y;
        int i1 = max(0, xi - 5);
        int i2 = min(xi + 5, height - 1);
        int j1 = max(0, yj - 5);
        int j2 = min(yj + 5, width - 1);

        double minvalue[3] = {1e10, 1e10, 1e10};
        Point * p = new Point[3];
        int num = 0;
        for (int k = i1; k <= i2; ++k)
        {
            for (int l = j1; l <= j2; ++l)
            {
                int temp = tri[k][l];

                if (temp == 0 || temp == 255)
                {
                    continue;
                }

                int index = unknownIndex[k][l];
                Tuple t   = tuples[index];
                if (t.flag == -1)
                {
                    continue;
                }

                double m  = mP(xi, yj, t.f, t.b);

                if (m > minvalue[2])
                {
                    continue;
                }

                if (m < minvalue[0])
                {
                    minvalue[2] = minvalue[1];
                    p[2]   = p[1];

                    minvalue[1] = minvalue[0];
                    p[1]   = p[0];

                    minvalue[0] = m;
                    p[0].x = k;
                    p[0].y = l;

                    ++num;

                }
                else if (m < minvalue[1])
                {
                    minvalue[2] = minvalue[1];
                    p[2]   = p[1];

                    minvalue[1] = m;
                    p[1].x = k;
                    p[1].y = l;

                    ++num;
                }
                else if (m < minvalue[2])
                {
                    minvalue[2] = m;
                    p[2].x = k;
                    p[2].y = l;

                    ++num;
                }
            }
        }

        num = min(num, 3);


        double fb = 0;
        double fg = 0;
        double fr = 0;
        double bb = 0;
        double bg = 0;
        double br = 0;
        double sf = 0;
        double sb = 0;

        for (int k = 0; k < num; ++k)
        {
            int i  = unknownIndex[p[k].x][p[k].y];
            fb += tuples[i].f.val[0];
            fg += tuples[i].f.val[1];
            fr += tuples[i].f.val[2];
            bb += tuples[i].b.val[0];
            bg += tuples[i].b.val[1];
            br += tuples[i].b.val[2];
            sf += tuples[i].sigmaf;
            sb += tuples[i].sigmab;
        }

        fb /= (num + 1e-10);
        fg /= (num + 1e-10);
        fr /= (num + 1e-10);
        bb /= (num + 1e-10);
        bg /= (num + 1e-10);
        br /= (num + 1e-10);
        sf /= (num + 1e-10);
        sb /= (num + 1e-10);

        Scalar fc = Scalar(fb, fg, fr);
        Scalar bc = Scalar(bb, bg, br);
        int b, g, r;
        b = data[xi * step +  yj* channels];
        g = data[xi * step +  yj * channels + 1];
        r = data[xi * step +  yj * channels + 2];
        Scalar pc = Scalar(b, g, r);
        double   df = distanceColor2(pc, fc);
        double   db = distanceColor2(pc, bc);
        Scalar tf = fc;
        Scalar tb = bc;

        int index = xi * width + yj;
        if (df < sf)
        {
            fc = pc;
        }
        if (db < sb)
        {
            bc = pc;
        }
        if (fc.val[0] == bc.val[0] && fc.val[1] == bc.val[1] && fc.val[2] == bc.val[2])
        {
            ftuples[index].confidence = 0.00000001;
        }
        else
        {
            ftuples[index].confidence = exp(-10 * mP(xi, yj, tf, tb));
        }


        ftuples[index].f = fc;
        ftuples[index].b = bc;


        ftuples[index].alphar = max(0.0, min(1.0,comalpha(pc, fc, bc)));
        //cvSet2D(matte, xi, yj, ScalarAll(ftuples[index].alphar * 255));
    }
    /*cvNamedWindow("1");
    cvShowImage("1", matte);*/
    /*cvNamedWindow("2");
    cvShowImage("2", trimap);*/
    /*cvWaitKey(0);*/
    tuples.clear();

}

void SharedMatting::localSmooth()
{
    vector<Point>::iterator it;
    double sig2 = 100.0 / (9 * 3.1415926);
    double r = 3 * sqrt(sig2);
    for (it = uT.begin(); it != uT.end(); ++it)
    {
        int xi = it->x;
        int yj = it->y;

        int i1 = max(0, int(xi - r));
        int i2 = min(int(xi + r), height - 1);
        int j1 = max(0, int(yj - r));
        int j2 = min(int(yj + r), width - 1);

        int indexp = xi * width + yj;
        Ftuple ptuple = ftuples[indexp];

        Scalar wcfsumup = Scalar::all(0);
        Scalar wcbsumup = Scalar::all(0);
        double wcfsumdown = 0;
        double wcbsumdown = 0;
        double wfbsumup   = 0;
        double wfbsundown = 0;
        double wasumup    = 0;
        double wasumdown  = 0;

        for (int k = i1; k <= i2; ++k)
        {
            for (int l = j1; l <= j2; ++l)
            {
                int indexq = k * width + l;
                Ftuple qtuple = ftuples[indexq];

                double d = dP(Point(xi, yj), Point(k, l));

                if (d > r)
                {
                    continue;
                }

                double wc;
                if (d == 0)
                {
                    wc = exp(-(d * d) / sig2) * qtuple.confidence;
                }
                else
                {
                    wc = exp(-(d * d) / sig2) * qtuple.confidence * abs(qtuple.alphar - ptuple.alphar);
                }
                wcfsumdown += wc * qtuple.alphar;
                wcbsumdown += wc * (1 - qtuple.alphar);

                wcfsumup.val[0] += wc * qtuple.alphar * qtuple.f.val[0];
                wcfsumup.val[1] += wc * qtuple.alphar * qtuple.f.val[1];
                wcfsumup.val[2] += wc * qtuple.alphar * qtuple.f.val[2];

                wcbsumup.val[0] += wc * (1 - qtuple.alphar) * qtuple.b.val[0];
                wcbsumup.val[1] += wc * (1 - qtuple.alphar) * qtuple.b.val[1];
                wcbsumup.val[2] += wc * (1 - qtuple.alphar) * qtuple.b.val[2];


                double wfb = qtuple.confidence * qtuple.alphar * (1 - qtuple.alphar);
                wfbsundown += wfb;
                wfbsumup   += wfb * sqrt(distanceColor2(qtuple.f, qtuple.b));

                double delta = 0;
                double wa;
                if (tri[k][l] == 0 || tri[k][l] == 255)
                {
                    delta = 1;
                }
                wa = qtuple.confidence * exp(-(d * d) / sig2) + delta;
                wasumdown += wa;
                wasumup   += wa * qtuple.alphar;
            }
        }

        int b, g, r;
        b = data[xi * step +  yj* channels];
        g = data[xi * step +  yj * channels + 1];
        r = data[xi * step +  yj * channels + 2];
        Scalar cp = Scalar(b, g, r);
        Scalar fp;
        Scalar bp;

        double dfb;
        double conp;
        double alp;

        bp.val[0] = min(255.0, max(0.0,wcbsumup.val[0] / (wcbsumdown + 1e-200)));
        bp.val[1] = min(255.0, max(0.0,wcbsumup.val[1] / (wcbsumdown + 1e-200)));
        bp.val[2] = min(255.0, max(0.0,wcbsumup.val[2] / (wcbsumdown + 1e-200)));

        fp.val[0] = min(255.0, max(0.0,wcfsumup.val[0] / (wcfsumdown + 1e-200)));
        fp.val[1] = min(255.0, max(0.0,wcfsumup.val[1] / (wcfsumdown + 1e-200)));
        fp.val[2] = min(255.0, max(0.0,wcfsumup.val[2] / (wcfsumdown + 1e-200)));

        //double tempalpha = comalpha(cp, fp, bp);
        dfb  = wfbsumup / (wfbsundown + 1e-200);

        conp = min(1.0, sqrt(distanceColor2(fp, bp)) / dfb) * exp(-10 * mP(xi, yj, fp, bp));
        alp  = wasumup / (wasumdown + 1e-200);

        double alpha_t = conp * comalpha(cp, fp, bp) + (1 - conp) * max(0.0, min(alp, 1.0));

        alpha[xi][yj] = alpha_t * 255;
    }
    ftuples.clear();
}
//存储图像
void SharedMatting::save(char * filename)
{

    imwrite(filename, matte);
}

void SharedMatting::getMatte()
{
    int h     = matte.rows;
    int w     = matte.cols;
    int s     = matte.step1();
    uchar* d  = (uchar *)matte.data;
    for (int i = 0; i < h; ++i)
    {
        for (int j = 0; j < w; ++j)
        {
            d[i*s+j] = alpha[i][j];

        }
    }
}

//主干方法
void SharedMatting::solveAlpha()
{
    clock_t start, finish;
    //expandKnown()
    start = clock();
    // cout << "Expanding..." << endl;
    expandKnown();
    // finish = clock();
    // cout <<  double(finish - start) / (CLOCKS_PER_SEC * 2.5) << endl;

    //gathering()
    // cout << "Gathering..." << endl;
    // start = clock();
    gathering();
    // finish = clock();
    // cout <<  double(finish - start) / (CLOCKS_PER_SEC * 2.5) << endl;

    //refineSample()
    // start = clock();
    // cout << "Refining..." << endl;
    refineSample();
    // finish = clock();
    // cout <<  double(finish - start) / (CLOCKS_PER_SEC * 2.5) << endl;

    //localSmooth()
    // start = clock();
    // cout << "LocalSmoothing..." << endl;
    localSmooth();
    finish = clock();
    cout << "solveAlpha Time: " << double(finish - start) / (CLOCKS_PER_SEC)*1000 << " ms" << endl;

    //getMatte()
    getMatte();
    // cvNamedWindow("matte");
    // cv::imshow("matte", matte);
    // cvWaitKey(0);
}
