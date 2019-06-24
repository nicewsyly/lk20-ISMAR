#ifndef _AFFINE_FA_H
#define _AFFINE_FA_H
#include <opencv2/opencv.hpp>
#include <iostream>
#include "tools.hpp"

cv::Matx<float,2,3> update_step(const cv::Matx<float,2,3>& warp_p,
                                cv::Mat delta_p);

//template 无法定义为返回值类型,因为无法推断,重载
template<typename T,int M,int N>
std::vector<fit> affine_fa(const cv::Mat& img,const cv::Mat& templ,
                           const cv::Matx<float,2,3>& p_init,const int n_iters,
                           const int step_size,cv::Matx<T,M,N>& warp_p,
                           const std::vector<cv::Point2f>& templ_pts)
{
    cv::Mat aimg;
    if(img.channels()==1)
        aimg=img.clone();
    else
        cv::cvtColor(img,aimg,cv::COLOR_BGR2GRAY);

    warp_p=p_init;
    const int height=templ.rows;
    const int width=templ.cols;
    std::vector<cv::Point2f> templ_contour={cv::Point2f(0,0),
                                            cv::Point2f(0,height),
                                            cv::Point2f(width,height),
                                            cv::Point2f(width,0)}; 
    //todo image gradient  dx ,dy    
    cv::Mat dx,dy;
    grad_a(aimg,dx,dy);
    //todo evaluate jacobian
    cv::Mat dwdp=jacobian_a(width,height);
    std::vector<fit> fita(n_iters);
    const int np=6;
    for(int ni=0;ni<n_iters;++ni)
    {
        //todo warp_a
        cv::Mat Iwxp=warp_a(img,warp_p,templ_pts);
        cv::Mat err_img=templ-Iwxp;
        fita[ni].warp_p=warp_p;
        //mean image val
        cv::Scalar mean_scalar=cv::mean(err_img.mul(err_img));
        float mean_s=mean_scalar.val[0];
        fita[ni].rms_err=std::sqrt(mean_s);
        
        if(ni==n_iters-1)
        {
            break;
        }
        cv::Mat nIx=warp_a(dx,warp_p,templ_pts);
        cv::Mat nIy=warp_a(dy,warp_p,templ_pts);
        //todo steest descent image
        cv::Mat vIdwdp=sd_images(dwdp,nIx,nIy,np,height,width);
        
        //todo hessian and iverse
        cv::Mat hess=hessian_a(vIdwdp,np,width);
        cv::Mat hess_inv=hess.inv();
        cv::Mat sd_delta_p=sd_update(vIdwdp,err_img,np,width);
        cv::Mat delta_p=hess_inv*sd_delta_p;
        warp_p=update_step(warp_p,delta_p);
        
         
    }
    return fita;
}

#endif
