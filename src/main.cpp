#include "saliency/saliency_rc.h"
#include "matting/saliency_cut.h"
#include "matting/sharedmatting.h"

enum TrimapValue {TrimapBackground = 0, TrimapUnknown = 128, TrimapForeground = 255};

void getTrimap(const cv::Mat &sal1f, cv::Mat& trimap, float t1, float t2)
{
    CV_Assert(sal1f.type() == CV_32FC1);

    int height = sal1f.rows; 
    int width = sal1f.cols;

    if(!trimap.data)
        trimap = cv::Mat::zeros(height, width, CV_8UC1);
    size_t n_unknown_pixels = 0;

    for (int y = 0; y < height; y++) {

        uchar* triVal = trimap.ptr<uchar>(y);
        const float *salVal = sal1f.ptr<float>(y);

        for (int x = 0; x < width; x++) {

            triVal[x] = salVal[x] < t1 ? TrimapBackground : TrimapUnknown;
            triVal[x] = salVal[x] > t2 ? TrimapForeground : triVal[x]; 
            if ( TrimapUnknown == triVal[x] )
                ++n_unknown_pixels;
        }
    }

    std::cout << "unknown pixels percent: " << (float)n_unknown_pixels/(height*width)*100 << "%" << std::endl;

    if ( n_unknown_pixels > size_t(0.05*height*width) ) 
        getTrimap( sal1f, trimap, t1+0.05, t2-0.05);
}


/**
 * A demo to show how to use the matting and saliency method
 *      alpha-matting
 *      saliencyCut
 * 
 *      RC
 *      FASA
 *      GMR
*/
int matting_demo(std::string root_dir, std::string salDir)
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

    clock_t start, finish;
    std::vector<cv::Mat> mv;

#pragma omp parallel for 
    /**
     * Salient Object Detection based on Global contrast
    */
    for (int i = 0; i < imgNum; i++){
        
        string image_path = paths[i];
        string saliency_map_path = salDir + names[i] + "_RCC.png";
        string origin_image_path = salDir + names[i] + ".png";
        cv::Mat img3f, img_half, triimg_half;

        if (FileOps::exists(saliency_map_path))
            continue;

        std::cout << "Processing " << i << "/" << imgNum << "th image: " << image_path << std::endl;

        start = clock();
        
        SharedMatting sm;
        cv::Mat img = cv::imread(image_path);
        cv::Mat sal, trimap, trimap3;
        CV_Assert_(img.data != NULL, ("Can't load image %s\n", image_path.c_str()));

        sm.setImage(img);
        img.convertTo(img3f, CV_32FC3, 1.0/255);

        /**
         *  FASA: Get Saliency Map
         */
        // get_saliency_map(img, sal);
        // sal.convertTo(sal, CV_32FC1, 1.0/255);

        /**
         *  GMR: Get Saliency Map
         */
        // GMRsaliency GMRsal;
        // sal_gmr = GMRsal.GetSal(img);
        // sal_gmr.convertTo(sal, CV_32FC1, 1.0/255);

        sal = CmSaliencyRC::GetRC(img3f);
        finish = clock();
        cout << "GetRC Time: " << double(finish - start) / (CLOCKS_PER_SEC)*1000 << " ms" << endl;
        getTrimap(sal, trimap, 0.3, 0.5);
        mv.clear();
        mv.push_back(trimap);
        mv.push_back(trimap);
        mv.push_back(trimap);
        cv::merge(mv, trimap3);
        // cv::resize(trimap3, triimg_half, cv::Size(img.cols/2, img.rows/2));

        sm.setTrimap(trimap3);
        sm.solveAlpha();
       
        // get saliency mask 
        cv::Mat cutMask;
        cutMask =  sm.getMask();
        cv::Mat foreground(img.size(), CV_8UC3, cv::Scalar(255, 255, 255));

        if (!cutMask.empty()) {

            img.copyTo(foreground, cutMask);
            finish = clock();
            cout << "Total Time: " << double(finish - start) / (CLOCKS_PER_SEC)*1000 << " ms" << endl;
            // cv::imwrite(origin_image_path, image);
            // cv::imwrite(saliency_map_path, foreground);
            cv::imshow("input", img);
            cv::imshow("trimap", trimap);
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


int main(int argc, char* argv[])
{
    if (argc != 2){
        std::cout << "Usage: saliency_cut input_dir" << std::endl;
        return 0;
    }

    std::string input_dir = argv[1];
    // Ensure directory end with "/"
    if(input_dir.back() != '/')
        input_dir += "/";
    std::string output_dir = input_dir + "saliency/";
    std::cout << "input_dir: " << input_dir << std::endl;
    std::cout << "output_dir: " << output_dir << std::endl;

    // Saliency detection methods and Matting methods
    matting_demo(input_dir, output_dir); 

    return 0;
}
