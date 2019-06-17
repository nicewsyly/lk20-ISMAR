#include <iostream>
#include <opencv2/opencv.hpp>
#include "affine_fa.hpp"

cv::Matx<float,2,3> update_step(const cv::Matx<float,2,3>& warp_p,cv::Mat delta_p)
{
    delta_p=delta_p.reshape(1,3).t();
    cv::Matx<float,2,3> result=warp_p+cv::Matx23f(delta_p);
        
    //return warp_p+cv::Matx23f(delta_p);
    return result;
}


