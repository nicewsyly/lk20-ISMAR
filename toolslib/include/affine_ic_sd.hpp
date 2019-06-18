#ifndef _AFFINE_ICSD_H
#define _AFFINE_ICSD_H
#include <opencv2/opencv.hpp>
#include <iostream>
#include "tools.hpp"

cv::Matx<float,2,3> icsdupdate_step(const cv::Matx<float,2,3>& warp_p,cv::Mat delta_p);
typedef struct icsdfit_
{
    cv::Matx23f warp_p;
    float rms_err;
}icsdfit;

//template 无法定义为返回值类型,因为无法推断,重载
template<typename T,int M,int N>
std::vector<icsdfit> affine_ic_sd(const cv::Mat& img,const cv::Mat& templ,const cv::Matx<float,2,3>& p_init,const int n_iters,const int step_size,cv::Matx<T,M,N>& warp_p,const std::vector<cv::Point2f>& templ_pts)
{
    cv::Mat aimg;
    if(img.channels()==1)
        aimg=img.clone();
    else
        cv::cvtColor(img,aimg,cv::COLOR_BGR2GRAY);
    warp_p=p_init;
    const int height=templ.rows;
    const int width=templ.cols;
    std::vector<cv::Point2f> templ_contour={cv::Point2f(0,0),cv::Point2f(0,height),cv::Point2f(width,height),cv::Point2f(width,0)}; 
    //todo image gradient  dx ,dy    
    cv::Mat nTx,nTy;
    grad_a(templ,nTx,nTy);
    
    const int np=6;
    //todo evaluate jacobian
    cv::Mat dwdp=jacobian_a(width,height);
    
    //todo compute steepest descent images
    cv::Mat vTdwdp=sd_images(dwdp,nTx,nTy,np,height,width);
    
    //compute hessian and inverse
    cv::Mat H=hessian_a(vTdwdp,np,width);
    cv::Mat dgdp2=H;
    //cv::Mat H_inv=H.inv();
    
    std::vector<icsdfit> fita(n_iters);
    for(int ni=0;ni<n_iters;++ni)
    {
        //todo warped images
        cv::Mat Iwxp=warp_a(img,warp_p,templ_pts);
        //todo err_imgs
        cv::Mat err_img=Iwxp-templ;
        fita[ni].warp_p=warp_p;
        //mean image val
        cv::Scalar mean_scalar=cv::mean(err_img.mul(err_img));
        float mean_s=mean_scalar.val[0];
        fita[ni].rms_err=std::sqrt(mean_s);
        
        if(ni==n_iters-1)
        {
            break;
        }
        cv::Mat dgdp=sd_update(vTdwdp,err_img,np,width).t();
        cv::Mat c;
        cv::Mat B=(dgdp*dgdp.t()).t();
        cv::Mat A=(dgdp*dgdp2*dgdp.t()).t();
        cv::solve(A,B,c);
         
        cv::Mat delta_p=c.t()*dgdp;
        warp_p=icupdate_step(warp_p,delta_p);
        
         
    }
    return fita;
}

#endif
