#ifndef _ISMARUTILS_H
#define _ISMARUTILS_H
#include <iostream>
#include <opencv2/opencv.hpp>

//gradient
void ismargrad_a(const cv::Mat& img,const cv::Mat& con_pts,cv::Mat& dx,cv::Mat& dy);
//jacobian 
cv::Mat ismarjacobian_a(const int width,const int height);
//warp I
cv::Mat ismarwarp_a(const cv::Mat& img,const cv::Matx23f& warp_p,
               const cv::Mat& con_pts);
// 
cv::Mat ismarsd_images(const cv::Mat& dwdp,const cv::Mat& nIx,const cv::Mat& nIy,
                  const int np,const int height,const int width);
//hessian
cv::Mat ismarhessian_a(const cv::Mat& vIdwdp,const int np,const int width);
//steepest descent 
cv::Mat ismarsd_update(const cv::Mat& vIdwdp,const cv::Mat& err_img,const int np,
                  const int w);

cv::Mat ismarquadtobox(const cv::Mat& img,const cv::Mat& pts,
                  const cv::Matx33f& M,const std::string& ftype);

std::vector<cv::Point2f> get_conpts(const cv::Mat& img,const std::vector<cv::Point2f>& contour);
#endif
