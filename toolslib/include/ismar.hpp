#ifndef _ISMAR_H
#define _ISMAR_H
#include <opencv2/opencv.hpp>
#include <iostream>
#include "ismar_tools.hpp"
#include "tools.hpp"

cv::Matx<float,2,3> ismarupdate_step(const cv::Matx<float,2,3>& warp_p,
                                cv::Mat delta_p);

std::vector<fit> ismar(const cv::Mat& firimg,const cv::Mat& secimg,
                           const std::vector<cv::Point2f> prevcontour,const cv::Matx23f& p_init);

#endif
