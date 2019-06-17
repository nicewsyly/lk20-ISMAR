#ifndef _UTILS_H
#define _UTILS_H
#include <iostream>
#include <opencv2/opencv.hpp>

void grad_a(const cv::Mat& img,cv::Mat& dx,cv::Mat& dy);
cv::Mat jacobian_a(const int width,const int height);
cv::Mat warp_a(const cv::Mat& img,const cv::Matx23f& warp_p,const std::vector<cv::Point2f>& templt_pts);
cv::Mat sd_images(const cv::Mat& dwdp,const cv::Mat& nIx,const cv::Mat& nIy,const int np,const int height,const int width);
cv::Mat hessian_a(const cv::Mat& vIdwdp,const int np,const int width);
cv::Mat sd_update(const cv::Mat& vIdwdp,const cv::Mat& err_img,const int np,const int w);
cv::Mat quadtobox(const cv::Mat& img,const std::vector<cv::Point2f>& pts,const cv::Matx33f& M,const std::string& ftype);
#endif
