#include <iostream>
#include <opencv2/opencv.hpp>
#include "affine_ic_lm.hpp"

cv::Matx<float,2,3> iclmupdate_step(const cv::Matx<float,2,3>& warp_p,cv::Mat delta_p)
{
    delta_p=delta_p.reshape(1,3).t();
    //cv::Matx<float,2,3> result=warp_p+cv::Matx23f(delta_p);
    cv::Matx13f bot(0,0,1);
    
    cv::Matx33f delta_M;
    cv::vconcat(delta_p,bot,delta_M);
    delta_M(0,0)+=1;
    delta_M(1,1)+=1;
    
    delta_M=delta_M.inv();
    
    
    cv::Matx33f warp_M;
    cv::vconcat(warp_p,bot,warp_M);
    warp_M(0,0)+=1;
    warp_M(1,1)+=1;
    
    cv::Matx33f comp_M=warp_M*delta_M;
    cv::Matx23f warp_pr=cv::Mat(comp_M).rowRange(cv::Range(0,2));
    warp_pr(0,0)-=1;
    warp_pr(1,1)-=1;
       
    //return warp_p+cv::Matx23f(delta_p);
    return warp_pr;
}


