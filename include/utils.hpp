#ifndef _MAIN_UTIL_H
#define _MAIN_UTIL_H
#include <opencv2/opencv.hpp>
#include <iostream>
#include <dirent.h>
#include <string>

#define DEBUG 1
#if DEBUG
#define P(logo,info) std::cout<<logo<<info<<std::endl;
#else
#define P(logo,info)
#endif

void get_videos( const std::string& path, const std::string& exd, std::vector<std::string>& files );
void get_contour(const cv::Mat& img,std::vector<cv::Point2f>& temp_contour);
cv::Mat DrawInlier(const cv::Mat &src1, const cv::Mat &src2, std::vector<cv::KeyPoint> &kpt1, std::vector<cv::KeyPoint> &kpt2, std::vector<cv::DMatch> &inlier, int type);
template<typename T, int M, int N>
void mat2pts(const cv::Matx<T,M,N>& mtx,std::vector<cv::Point2f>& pts)
{
    pts.clear();
    const int rows=M;
    const int cols=N;
    for(unsigned int ci=0;ci<cols;++ci)
    {
        pts.push_back(cv::Point2f(mtx(0,ci),mtx(1,ci)));
    }
}
#endif
