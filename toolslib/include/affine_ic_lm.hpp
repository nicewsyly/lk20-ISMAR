#ifndef _AFFINE_ICLM_H
#define _AFFINE_ICLM_H
#include <opencv2/opencv.hpp>
#include <iostream>
#include "tools.hpp"

cv::Matx<float,2,3> iclmupdate_step(const cv::Matx<float,2,3>& warp_p,cv::Mat delta_p);
/*
typedef struct iclmfit_
{
    cv::Matx23f warp_p;
    float rms_err;
}iclmfit;
*/

//template 无法定义为返回值类型,因为无法推断,重载
template<typename T,int M,int N>
std::vector<fit> affine_ic_lm(const cv::Mat& img,const cv::Mat& templ,const cv::Matx<float,2,3>& p_init,const int n_iters,const int step_size,cv::Matx<T,M,N>& warp_p,const std::vector<cv::Point2f>& templ_pts)
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

    //
    float delta=0.001;
    cv::Mat Iwxp=warp_a(img,warp_p,templ_pts);

    //todo compute error image
    cv::Mat err_img=Iwxp-templ;
    double e=std::sqrt(cv::mean(err_img.mul(err_img)).val[0]); 

    //todo image gradient  nIx , nIy    
    cv::Mat nTx,nTy;
    grad_a(templ,nTx,nTy);
    
    const int np=6;
    //todo evaluate jacobian
    cv::Mat dwdp=jacobian_a(width,height);
    
    //todo compute steepest descent images
    cv::Mat vTdwdp=sd_images(dwdp,nTx,nTy,np,height,width);
    
    //compute hessian and inverse
    cv::Mat H=hessian_a(vTdwdp,np,width);
   
    //LM matrix
    cv::Mat LM=cv::Mat::zeros(cv::Size(np,np),CV_32FC1);
    for(int i=0;i<np;++i)
    {
	cv::Mat h1=vTdwdp.colRange(cv::Range(i*width+1,i*width+width));
	LM.at<float>(i,i)=cv::sum(h1.mul(h1)).val[0];
    }
     
    std::vector<fit> fita(n_iters);
    int ni=0;
    fita[ni].warp_p=warp_p;
    fita[ni].rms_err=e;
    for(int ni=1;ni<n_iters;++ni)
    {
        cv::Mat sd_delta_p=sd_update(vTdwdp,err_img,np,width);
        cv::Mat H_lm=H+delta*LM;
        cv::Mat H_inv=H_lm.inv();
        
        cv::Mat delta_p=H_inv*sd_delta_p;
        cv::Matx23f warp_p_lm=iclmupdate_step(warp_p,delta_p);
        
        cv::Mat Iwxp=warp_a(img,warp_p_lm,templ_pts);
        
        cv::Mat err_img_lm=Iwxp-templ;
        double err_lm=std::sqrt(cv::mean(err_img_lm.mul(err_img_lm)).val[0]); 

        if (e<err_lm)
        {
            delta*=10;
        }
        else
        {
            delta/=10;
            err_img=err_img_lm;
            e=err_lm;
            warp_p=warp_p_lm;
        }
        fita[ni].warp_p=warp_p;
        fita[ni].rms_err=e;
        if (ni==n_iters)
            break;
         
    }
    return fita;
}

#endif
