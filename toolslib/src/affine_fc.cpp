#include "affine_fc.hpp"
cv::Matx<float,2,3> fcupdate_step(const cv::Matx<float,2,3>& warp_p,cv::Mat delta_p)
{
    delta_p=delta_p.reshape(1,3).t();
    cv::Matx33f delta_M;
    cv::Mat bot=(cv::Mat_<float>(1,3)<<0,0,1);
    cv::vconcat(delta_p,bot,delta_M); 
    delta_M(0,0)+=1;
    delta_M(1,1)+=1;
    
    cv::Matx33f warp_M;
    cv::vconcat(warp_p,bot,warp_M);
    warp_M(0,0)+=1;
    warp_M(1,1)+=1;
    
    cv::Matx33f comp_M=warp_M*delta_M;
    cv::Matx23f warp_pt=cv::Mat(comp_M).rowRange(cv::Range(0,2));
    warp_pt(0,0)-=1;
    warp_pt(1,1)-=1;
    return warp_pt;
}
