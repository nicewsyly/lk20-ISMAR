#ifndef _AFFINE_ICD_H
#define _AFFINE_ICD_H
#include <opencv2/opencv.hpp>
#include <iostream>
#include "tools.hpp"

cv::Matx<float,2,3> icdupdate_step(const cv::Matx<float,2,3>& warp_p,cv::Mat delta_p);
typedef struct icdfit_
{
    cv::Matx23f warp_p;
    float rms_err;
}icdfit;

//template 无法定义为返回值类型,因为无法推断,重载
template<typename T,int M,int N>
std::vector<icdfit> affine_ic_d(const cv::Mat& img,const cv::Mat& templ,const cv::Matx<float,2,3>& p_init,const int n_iters,const int step_size,cv::Matx<T,M,N>& warp_p,const std::vector<cv::Point2f>& templ_pts)
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
    //affine_ic:
    //cv::Mat H=hessian_a(vTdwdp,np,width);
    //cv::Mat H_inv=H.inv();
    //affine_ic_d:
    cv::Mat H,Hg;
    if(step_size)
    {
        Hg=hessian_a(vTdwdp,np,width);
        H=cv::Mat::diag(Hg.diag(0));
    }
    else
    {
        H=cv::Mat::zeros(cv::Size(np,np),CV_32FC1);
        for(int i=0;i<np;++i)
        {
            cv::Mat h1=vTdwdp.colRange(cv::Range(i*width+1,i*width+width));
            H.at<float>(i,i)=cv::sum(h1.mul(h1)).val[0];
        }
    }
    cv::Mat H_inv=H.inv();
    std::vector<icdfit> fita(n_iters);
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
        cv::Mat sd_delta_p=sd_update(vTdwdp,err_img,np,width);
        
        cv::Mat delta_p=H_inv*sd_delta_p;
        if(step_size)
        {
            cv::Mat c;
            cv::Mat B=((sd_delta_p.t())*delta_p).t();
            cv::Mat A=(delta_p.t()*Hg*delta_p).t();
            cv::solve(A,B,c);
            delta_p=c.t()*delta_p;
        }
        else
            delta_p=H_inv*sd_delta_p;
        warp_p=icdupdate_step(warp_p,delta_p);
        
         
    }
    return fita;
}

#endif
