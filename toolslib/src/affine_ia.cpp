#include "affine_ia.hpp"

cv::Matx<float,2,3> iaupdate_step(const cv::Matx<float,2,3>& warp_p,cv::Mat delta_p_star)
{
    cv::Matx22f ss=cv::Mat(warp_p).colRange(cv::Range(0,2));
    ss(0,0)+=1;
    ss(1,1)+=1;
    
    cv::Mat eye_mat=cv::Mat::eye(3,3,CV_32F);
    //cv::Mat sigma_p_inv(cv::Size(eye_mat.cols*ss.cols,eye_mat.rows*ss.rows),CV_32F);
    cv::Mat sigma_p_inv;
    for(int eri=0;eri<eye_mat.rows;++eri)
    {
        cv::Mat tmp_;
        float* data=eye_mat.ptr<float>(eri,0);
        for(int eci=0;eci<eye_mat.cols;++eci)
        {
            if(eci==0)
                tmp_=data[eci]*cv::Mat(ss).clone();
            else
                cv::hconcat(tmp_,data[eci]*ss,tmp_);
        }
        if(eri==0)
            sigma_p_inv=tmp_.clone();
        else
            cv::vconcat(sigma_p_inv,tmp_,sigma_p_inv);
    }
    
    cv::Mat delta_p=(sigma_p_inv*delta_p_star);
    delta_p=delta_p.reshape(1,3).t();
    cv::Matx23f warp_pt=warp_p-cv::Matx23f(delta_p);
    return warp_pt;
}
