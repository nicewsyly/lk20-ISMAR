#ifndef _AFFINE_ICNT_H
#define _AFFINE_ICNT_H
#include <opencv2/opencv.hpp>
#include <iostream>
#include "tools.hpp"

cv::Matx<float,2,3> icntupdate_step(const cv::Matx<float,2,3>& warp_p,cv::Mat delta_p);
typedef struct icntfit_
{
    cv::Matx23f warp_p;
    float rms_err;
}icntfit;

//template 无法定义为返回值类型,因为无法推断,重载
template<typename T,int M,int N>
std::vector<icntfit> affine_ic_nt(const cv::Mat& img,const cv::Mat& templ,const cv::Matx<float,2,3>& p_init,const int n_iters,const int step_size,cv::Matx<T,M,N>& warp_p,const std::vector<cv::Point2f>& templ_pts)
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
    cv::Mat dxx,dxy,dyx,dyy;
    grad_a(templ,nTx,nTy);
    grad_a(nTx,dxx,dxy);
    grad_a(nTy,dyx,dyy);
    
    const int np=6;
    //todo evaluate jacobian
    cv::Mat dwdp=jacobian_a(width,height);
    
    //todo compute steepest descent images
    cv::Mat vTdwdp=sd_images(dwdp,nTx,nTy,np,height,width);

    //compute H_extra
    cv::Mat H_extra=cv::Mat::zeros(cv::Size(np*width,np*height),CV_32F);
    for(int i=0;i<np;++i)
    {
        cv::Mat tx=dwdp(cv::Rect(i*width,0,width,height));
        cv::Mat ty=dwdp(cv::Rect(i*width,height,width,dwdp.rows-height));
        cv::Mat tx2=(tx.mul(dxx)+ty.mul(dxy));
        cv::Mat ty2=(tx.mul(dxy)+ty.mul(dyy));
        for(int j=0;j<np;++j)
        {
            cv::Mat qx=dwdp(cv::Rect(j*width,0,width,height));
            cv::Mat qy=dwdp(cv::Rect(j*width,height,width,dwdp.rows-height));
            H_extra(cv::Rect(j*width,i*height,width,height))=tx2.mul(qx)+ty2.mul(qy);
        }
    } 
    
    //compute hessian and inverse
    cv::Mat H=hessian_a(vTdwdp,np,width);
    
    std::vector<icntfit> fita(n_iters);
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
        //compute hessian and inverse
        cv::Mat H_nt=cv::Mat::zeros(np,np,CV_32F);
        for(int i=0;i<np;++i)
        {
            for(int j=0;j<np;++j)
            {
                cv::Mat sub_h=H_extra(cv::Rect(j*width,i*height,width,height)).clone();
                cv::Mat mul_h=(sub_h.mul(err_img)); 
                H_nt.at<float>(i,j)=cv::sum(mul_h).val[0];
            }
        }
        H_nt+=H;
        cv::Mat H_inv=H_nt.inv();
        cv::Mat dgdp=sd_update(vTdwdp,err_img,np,width);
        
        cv::Mat delta_p=H_inv*dgdp;
        warp_p=icntupdate_step(warp_p,delta_p);
        
         
    }
    return fita;
}

#endif
